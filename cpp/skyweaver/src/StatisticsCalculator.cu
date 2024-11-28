#include "psrdada_cpp/cuda_utils.hpp"
#include "skyweaver/StatisticsCalculator.cuh"
#include "thrust/host_vector.h"

#include <cstring>
#include <ctime>
#include <errno.h>
#include <fstream>
#include <iostream>
#include <numeric>
#include <sstream>

#define LOG2_SKYWEAVER_NSAMPLES_PER_HEAP 8

namespace skyweaver
{
namespace kernel
{

__device__ void accumulate(double power,
                           long long& n,
                           double& M1,
                           double& M2,
                           double& M3,
                           double& M4)
{
    long long n1 = n;
    n++;
    double delta    = power - M1;
    double delta_n  = delta / n;
    double delta_n2 = delta_n * delta_n;
    double term1    = delta * delta_n * n1;
    M1 += delta_n;
    M4 += term1 * delta_n2 * (n * n - 3 * n + 3) + 6 * delta_n2 * M2 -
          4 * delta_n * M3;
    M3 += term1 * delta_n * (n - 2) - 3 * delta_n * M2;
    M2 += term1;
}

/**
 * @brief Calculate statistics for the given input data
 *
 * @param ftpa_voltages Voltages data in FTPA order (8,8-bit complex)
 * @param results An array of statistics objects to store the outputs
 * @param nsamples The total number of time samples in the T dimension
 *
 * @details The FTPA order input allows for coalesced reads for the input
 *          with the caveat that we are only reading 16 bits at a time.
 *          The output is currently stored as an array of structs. A struct
 *          of arrays would likely be more performant here but is uglier
 *          from a coding persepective.
 *
 */
__global__ void calculate_statistics(char2 const* __restrict__ ftpa_voltages,
                                     Statistics* __restrict__ results,
                                     int nsamples)
{
    // Will operate on FTPA data and calculate statistics for FPA
    const int channel_idx = blockIdx.x;
    const int pol_idx     = blockIdx.y;
    const int antenna_idx = threadIdx.x;
    const int npol        = gridDim.y;
    const int nantennas   = blockDim.x;
    const int tpa_size    = npol * nantennas * nsamples;
    const int offset =
        channel_idx * tpa_size + pol_idx * nantennas + antenna_idx;
    const int stride = npol * nantennas;
    double M1 = 0.0, M2 = 0.0, M3 = 0.0, M4 = 0.0;
    long long n = 0;
    for(int sample_idx = offset; sample_idx < (tpa_size + offset);
        sample_idx += stride) {
        char2 data = ftpa_voltages[sample_idx];
        accumulate(static_cast<double>(data.x), n, M1, M2, M3, M4);
        accumulate(static_cast<double>(data.y), n, M1, M2, M3, M4);
    }

    // Output is ordered in FPA order
    int output_idx =
        channel_idx * npol * nantennas + pol_idx * nantennas + antenna_idx;
    Statistics* output = &results[output_idx];
    output->mean       = static_cast<float>(M1);
    output->std        = static_cast<float>(sqrt(M2 / (n - 1.0)));
    output->skew     = static_cast<float>(sqrt((double)n) * M3 / pow(M2, 1.5));
    output->kurtosis = static_cast<float>((double)n * M4 / (M2 * M2) - 3.0);
}

} // namespace kernel

StatisticsCalculator::StatisticsCalculator(PipelineConfig const& config,
                                           cudaStream_t stream)
    : _config(config), _stream(stream)
{
    BOOST_LOG_TRIVIAL(debug)
        << "Constructing new StatisticsCalculator instance";
}

StatisticsCalculator::~StatisticsCalculator()
{
    if(_stats_file.is_open()) {
        _stats_file.close();
    }
}

void StatisticsCalculator::calculate_statistics(
    FTPAVoltagesD<char2> const& ftpa_voltages)
{
    _stats_d.resize({ftpa_voltages.nchannels(),
                     ftpa_voltages.npol(),
                     ftpa_voltages.nantennas()});
    _stats_d.metalike(ftpa_voltages);
    _stats_d.tsamp(ftpa_voltages.tsamp() * ftpa_voltages.nsamples());
    _stats_h.like(_stats_d);

    int nsamples = ftpa_voltages.nsamples();
    // call kernel
    char2 const* ftpa_voltages_ptr =
        thrust::raw_pointer_cast(ftpa_voltages.data());
    Statistics* stats_ptr = thrust::raw_pointer_cast(_stats_d.data());
    dim3 dimBlock(_stats_d.nantennas());
    dim3 dimGrid(_stats_d.nchannels(), _stats_d.npol());
    kernel::calculate_statistics<<<dimGrid, dimBlock, 0, _stream>>>(
        ftpa_voltages_ptr,
        stats_ptr,
        nsamples);
    CUDA_ERROR_CHECK(cudaStreamSynchronize(_stream));
    BOOST_LOG_TRIVIAL(debug) << "Finished running calculate_statistics kernel";
    // Copy statistics to host
    _stats_h = _stats_d;
    BOOST_LOG_TRIVIAL(debug) << "Copied input levels to host";
}

void StatisticsCalculator::update_scalings(
    BeamsetWeightsVectorTypeH const& beamset_weights,
    int nbeamsets)
{
    // At this stage we have the standard deviations of each channel
    // available on the host (h_input_levels) To support post-fact rescaling
    // of the data it is the scales and offsets that must be preserved to
    // disk.
    const float weights_amp = 127.0f;

    if ((_config.ib_fscrunch() != _config.cb_fscrunch()) || 
        (_config.ib_tscrunch() != _config.cb_tscrunch())) {
        throw std::invalid_argument("IB and CB must share same F and T scrunch");
    }
    const std::size_t output_nchans = _config.nchans() / _config.cb_fscrunch();
    const std::size_t fscrunch = _config.cb_fscrunch();
    const std::size_t tscrunch = _config.cb_tscrunch();

    // Offsets for the coherent beams
    _cb_offsets_d.resize(output_nchans * nbeamsets);
    _cb_offsets_h.resize(output_nchans * nbeamsets);

    // Scalings for the coherent beams
    _cb_scaling_d.resize(output_nchans * nbeamsets);
    _cb_scaling_h.resize(output_nchans * nbeamsets);

    // Offsets for the incoherent beam
    _ib_offsets_d.resize(output_nchans * nbeamsets);
    _ib_offsets_h.resize(output_nchans * nbeamsets);

    // Scalings for the incoherent beam
    _ib_scaling_d.resize(output_nchans * nbeamsets);
    _ib_scaling_h.resize(output_nchans * nbeamsets);

    const std::uint32_t pa = _config.npol() * _config.nantennas();
    const std::uint32_t a  = _config.nantennas();

    for(std::uint32_t beamset_idx = 0; beamset_idx < nbeamsets; ++beamset_idx) {

        // Here we compute the offsets and scaling factors for I, Q, U and V for each beamset
        // Statistics are in FPA order

        for(int fo_idx = 0; fo_idx < output_nchans; ++fo_idx) {

            const int output_idx = beamset_idx * nbeamsets + fo_idx;

            // reset main accumulators
            float meanI  = 0.0f;
            float meanQ  = 0.0f;
            float varIQc = 0.0f;
            float varUVc = 0.0f;
            float varIQi = 0.0f;
            float varUVi = 0.0f;

            for(int fs_idx = 0; fs_idx < fscrunch; ++fs_idx) {
                const int f_idx = fo_idx * fscrunch + fs_idx;

                // Per antenna/pol accumulators
                float var_p0_sum = 0.0f; // sum(sigma_p0^2)
                float var_p1_sum = 0.0f; // sum(sigma_p1^2)
                float quad_sum = 0.0f;   // sum(sigma_p0^4 + sigma_p1^4)
                float sum_mul = 0.0f;    // sum(sigma_p0^2 * sigma_p1^2)
                for (int a_idx = 0; a_idx < _cofig.npol(); ++a_idx) {
                    // stats are in FPA order
                    // weights are in FPBA order
                    const int input_idx = f_idx * pa + a_idx;
                    Statistics p0 = _stats_h[input_idx];
                    Statistics p1 = _stats_h[input_idx + nantennas];
                    const float weight = beamset_weights[beamset_idx * a + a_idx];
                    const float std_p0 = p0.std * weight;
                    const float std_p1 = p1.std * weight;
                    const float var_p0 = std_p0 * std_p0;
                    const float var_p1 = std_p1 * std_p1;
                    meanI += var_p0 + var_p1;
                    meanQ += var_p0 - var_p1;
                    var_p0_sum += var_p0;
                    var_p1_sum += var_p1;
                    quad_sum += (var_p0 * var_p0) + (var_p1 * var_p1);
                    sum_mul += var_p0 * var_p1;
                }
                varIQc += (var_p0_sum * var_p0_sum) + (var_p1_sum * var_p1_sum);
                varIQi += quad_sum;
                varUVc += var_p0_sum * var_p1_sum;
                varUVi += sum_mul;
            }

            // Coherent offsets
            _cb_offsets_h[output_idx].x = 2 * tscrunch * meanI;  // I
            _cb_offsets_h[output_idx].y = 2 * tscrunch * meanQ;  // Q
            _cb_offsets_h[output_idx].z = 0.0f;                  // U
            _cb_offsets_h[output_idx].w = 0.0f;                  // V

            // Coherent scales
            _cb_scaling_h[output_idx].x = 4 * tscrunch * varIQc / _config.output_level(); // I
            _cb_scaling_h[output_idx].y = 4 * tscrunch * varIQc / _config.output_level(); // Q
            _cb_scaling_h[output_idx].z = 8 * tscrunch * varUVc / _config.output_level(); // U
            _cb_scaling_h[output_idx].w = 8 * tscrunch * varUVc / _config.output_level(); // V

            // Incoherent offsets
            _ib_offsets_h[output_idx].x = 2 * tscrunch * meanI;  // I
            _ib_offsets_h[output_idx].y = 2 * tscrunch * meanQ;  // Q
            _ib_offsets_h[output_idx].z = 0.0f;                  // U
            _ib_offsets_h[output_idx].w = 0.0f;                  // V

            // Incoherent scales
            _ib_scaling_h[output_idx].x = 4 * tscrunch * varIQi / _config.output_level(); // I
            _ib_scaling_h[output_idx].y = 4 * tscrunch * varIQi / _config.output_level(); // Q
            _ib_scaling_h[output_idx].z = 8 * tscrunch * varUVi / _config.output_level(); // U
            _ib_scaling_h[output_idx].w = 8 * tscrunch * varUVi / _config.output_level(); // V
        }
    }
    _cb_offsets_d = _cb_offsets_h;
    _cb_scaling_d = _cb_scaling_h;
    _ib_offsets_d = _ib_offsets_h;
    _ib_scaling_d = _ib_scaling_h;
}

void StatisticsCalculator::dump_all_scalings() const
{
    // Data can be written to the same place as the transient dumps
    // Total volume will typically by 2048 * 4 * 2 * 2 = 32 kB per target
    std::time_t now = std::time(0);
    std::tm* now_tm = std::gmtime(&now);
    char timestamp[42];
    std::strftime(timestamp, 42, "%Y-%m-%dT%X", now_tm);
    dump_scalings(timestamp, "cb_offsets", _config.output_dir(), _cb_offsets_h);
    dump_scalings(timestamp, "cb_scaling", _config.output_dir(), _cb_scaling_h);
    dump_scalings(timestamp, "ib_offsets", _config.output_dir(), _ib_offsets_h);
    dump_scalings(timestamp, "ib_scaling", _config.output_dir(), _ib_scaling_h);
}

void StatisticsCalculator::dump_scalings(
    std::string const& timestamp,
    std::string const& tag,
    std::string const& path,
    ScalingVectorTypeH const& ar) const
{
    std::ofstream writer;
    std::string filename = path + "/" + timestamp + "_" + tag + "_" +
                           std::to_string(_config.centre_frequency()) +
                           "Hz.bin";
    writer.open(filename, std::ofstream::out | std::ofstream::binary);
    if(writer.is_open()) {
        BOOST_LOG_TRIVIAL(info) << "Opened output file " << filename;
    } else {
        std::stringstream error_message;
        error_message << "Could not open file " << filename;
        BOOST_LOG_TRIVIAL(error) << error_message.str();
        return;
    }
    writer.write(static_cast<char*>(ar.data()), ar.size() * sizeof(ScalingType));
    writer.close();
}

StatisticsCalculator::StatisticsVectorTypeD const&
StatisticsCalculator::statistics() const
{
    return _stats_d;
}

StatisticsCalculator::ScalingVectorTypeD const&
StatisticsCalculator::cb_offsets() const
{
    return _cb_offsets_d;
}

StatisticsCalculator::ScalingVectorTypeD const&
StatisticsCalculator::cb_scaling() const
{
    return _cb_scaling_d;
}

StatisticsCalculator::ScalingVectorTypeD const&
StatisticsCalculator::ib_offsets() const
{
    return _ib_offsets_d;
}

StatisticsCalculator::ScalingVectorTypeD const&
StatisticsCalculator::ib_scaling() const
{
    return _ib_scaling_d;
}

} // namespace skyweaver