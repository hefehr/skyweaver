#include "cuComplex.h"
#include "psrdada_cpp/cuda_utils.hpp"
#include "skyweaver/CoherentBeamformer.cuh"

#include <cassert>

namespace skyweaver
{
namespace kernels
{

template <typename BfTraits>
__global__ void bf_ftpa_general_k(
    int2 const* __restrict__ ftpa_voltages,
    int2 const* __restrict__ fbpa_weights,
    typename BfTraits::QuantisedPowerType* __restrict__ tfb_powers,
    float const* __restrict__ output_scale,
    float const* __restrict__ output_offset,
    int const* __restrict__ beamset_mapping,
    typename BfTraits::RawPowerType const* __restrict__ ib_powers,
    int nsamples)
{
    /**
     * Perform compile time checks on requested beamforming parameters.
     */
    static_assert(SKYWEAVER_NBEAMS % SKYWEAVER_CB_WARP_SIZE == 0,
                  "Kernel can only process a multiple of 32 beams.");
    static_assert(SKYWEAVER_CB_NTHREADS % SKYWEAVER_CB_WARP_SIZE == 0,
                  "Number of threads must be an integer multiple of "
                  "SKYWEAVER_CB_WARP_SIZE.");
    static_assert(SKYWEAVER_NANTENNAS % 4 == 0,
                  "Number of antennas must be a multiple of 4.");
    static_assert(SKYWEAVER_NPOL == 2,
                  "This kernel only works for dual polarisation data.");

    /**
     * Allocated shared memory to store beamforming weights and temporary space
     * for antenna data.
     */
    __shared__ int2
        shared_apb_weights[SKYWEAVER_NANTENNAS / 4][SKYWEAVER_CB_WARP_SIZE];
    __shared__ int2
        shared_antennas[SKYWEAVER_CB_NTHREADS / SKYWEAVER_CB_WARP_SIZE]
                       [SKYWEAVER_NANTENNAS / 4];
    int const warp_idx = threadIdx.x / 0x20;
    int const lane_idx = threadIdx.x & 0x1f;
    /**
     * Each warp processes 32 beams (i.e. one beam per lane).
     */
    int const start_beam_idx = blockIdx.z * SKYWEAVER_CB_WARP_SIZE;

    /**
     * Complex multiply accumulators
     */
    int xx, yy, xy, yx;
    typename BfTraits::RawPowerType power = BfTraits::zero_power;
    int2 antennas, weights;
    int antenna_group_idx;
    const int sample_offset =
        SKYWEAVER_IB_TSCRUNCH *
        (blockIdx.x * SKYWEAVER_CB_NWARPS_PER_BLOCK + warp_idx);

    for(int channel_idx = blockIdx.y * SKYWEAVER_CB_FSCRUNCH;
        channel_idx < (blockIdx.y + 1) * SKYWEAVER_CB_FSCRUNCH;
        ++channel_idx) {
        /**
         * Here we load all the beamforming weights neccessary for this block.
         * Implicit assumption here is that we do not need to change the weights
         * over the timescale of the data processed in one block. This is almost
         * certainly OK if the input data has already been rotated to telescope
         * boresight and we are only applying parallactic angle tracking
         * updates.
         *
         * The global load is coalesced 8-byte (vectorised int2).
         */
        int const fbpa_weights_offset =
            SKYWEAVER_NANTENNAS / 4 *
            (SKYWEAVER_NBEAMS * channel_idx + (start_beam_idx + warp_idx));
        for(antenna_group_idx = lane_idx;
            antenna_group_idx < SKYWEAVER_NANTENNAS / 4;
            antenna_group_idx += SKYWEAVER_CB_WARP_SIZE) {
            shared_apb_weights[antenna_group_idx][warp_idx] = int2_transpose(
                fbpa_weights[fbpa_weights_offset + antenna_group_idx]);
        }

        // wait for all weights to load.
        __syncthreads();

        /**
         * Below is the main loop of the kernel. Here the kernel reads all the
         * antennas for a given sample and computes 32 beams. Each thread
         * computes only 1 beam and access to all the antennas required for that
         * computation is achieved via a shared memory broadcasts.
         */
        for(int sample_idx = sample_offset;
            sample_idx < (sample_offset + SKYWEAVER_IB_TSCRUNCH);
            ++sample_idx) {
            int ftpa_voltages_partial_idx =
                SKYWEAVER_NANTENNAS / 4 * SKYWEAVER_NPOL *
                (nsamples * channel_idx + sample_idx);

            // The loop below will compute the voltages for one pol/chan/sample
            // To make stokes vectors we need to keep this voltage in memory
            // to allow for the combination of the two polarisations.
            cuFloatComplex pol_voltage[SKYWEAVER_NPOL];
            for(int pol_idx = 0; pol_idx < SKYWEAVER_NPOL; ++pol_idx) {
                // Set the complex accumulator to zero before adding the next
                // polarisation
                xx = 0;
                yy = 0;
                xy = 0;
                yx = 0;

                /**
                 * Load all antennas antennas required for this sample into
                 * shared memory.
                 */
                if(lane_idx < SKYWEAVER_NANTENNAS / 4) {
                    shared_antennas[warp_idx][lane_idx] = int2_transpose(
                        ftpa_voltages[ftpa_voltages_partial_idx + lane_idx +
                                      SKYWEAVER_NANTENNAS / 4 * pol_idx]);
                }
                // Wait for all lanes to complete the load.
                __threadfence_block();
                for(antenna_group_idx = 0;
                    antenna_group_idx < SKYWEAVER_NANTENNAS / 4;
                    ++antenna_group_idx) {
                    // broadcast load 4 antennas
                    antennas = shared_antennas[warp_idx][antenna_group_idx];
                    // load corresponding 4 weights
                    weights = shared_apb_weights[antenna_group_idx][lane_idx];
                    // dp4a multiply add
                    dp4a(xx, weights.x, antennas.x);
                    dp4a(yy, weights.y, antennas.y);
                    dp4a(xy, weights.x, antennas.y);
                    dp4a(yx, weights.y, antennas.x);
                }
                pol_voltage[pol_idx].x = (float)xx - (float)yy; // real
                pol_voltage[pol_idx].y = (float)xy + (float)yx; // imag
            }
            BfTraits::integrate_stokes(pol_voltage[0], pol_voltage[1], power);
        }
        __syncthreads();
    }
    int const beam_idx          = (start_beam_idx + lane_idx);
    int const output_sample_idx = sample_offset / SKYWEAVER_CB_TSCRUNCH;
    int const nsamps_out        = nsamples / SKYWEAVER_CB_TSCRUNCH;


    // Specify TFB order outputs
    int const output_idx = output_sample_idx * gridDim.y * SKYWEAVER_NBEAMS // T
                           + blockIdx.y * SKYWEAVER_NBEAMS                  // F
                           + beam_idx;                                      // B

    int const beamset_idx  = beamset_mapping[beam_idx];
    int const ib_power_idx = beamset_idx * nsamps_out * gridDim.y +
                             output_sample_idx * gridDim.y + blockIdx.y;
    int const scloff_idx = beamset_idx * gridDim.y + blockIdx.y;
    float scale          = output_scale[scloff_idx];
    typename BfTraits::RawPowerType ib_power = ib_powers[ib_power_idx];
#if SKYWEAVER_IB_SUBTRACTION
    /*
    Because we inflate the weights to have a magnitude of 127 to make sure
    that they can still represent many phases, we also need to account for
    this scaling factor in the incoherent beam.
    */
    typename BfTraits::RawPowerType power_fp32 =
        BfTraits::ib_subtract(power, ib_power, 16129.0f, scale);
#else
    typename BfTraits::RawPowerType power_fp32 =
        BfTraits::rescale(power, output_offset[scloff_idx], scale);
#endif // SKYWEAVER_IB_SUBTRACTION
    tfb_powers[output_idx] = BfTraits::clamp(power_fp32);
}

} // namespace kernels

template <typename BfTraits>
CoherentBeamformer<BfTraits>::CoherentBeamformer(PipelineConfig const& config)
    : _config(config)
{
    BOOST_LOG_TRIVIAL(debug) << "Constructing CoherentBeamformer instance";

}

template <typename BfTraits>
CoherentBeamformer<BfTraits>::~CoherentBeamformer()
{
}

template <typename BfTraits>
void CoherentBeamformer<BfTraits>::beamform(
    VoltageVectorTypeD const& input,
    WeightsVectorTypeD const& weights,
    ScalingVectorTypeD const& output_scale,
    ScalingVectorTypeD const& output_offset,
    MappingVectorTypeD const& beamset_mapping,
    RawPowerVectorTypeD const& ib_powers,
    PowerVectorTypeD& output,
    int nbeamsets,
    cudaStream_t stream)
{
    if(output_scale.size() !=
        _config.nchans() / _config.cb_fscrunch() * nbeamsets) {
        throw std::runtime_error(
            "Unexpected number of channels in scaling vector");
    }
    if(output_offset.size() !=
        _config.nchans() / _config.cb_fscrunch() * nbeamsets) {
        throw std::runtime_error(
            "Unexpected number of channels in offset vector ");
    }
    if(beamset_mapping.size() != _config.nbeams()) {
        throw std::runtime_error("Unexpected size of beamset_mapping vector");
    }
    // First work out nsamples and resize output if not done already
    BOOST_LOG_TRIVIAL(debug) << "Executing coherent beamforming";
    std::size_t nsamples = input.nsamples();
    std::size_t output_size =
        (input.size() / _config.nantennas() / _config.npol() /
         _config.cb_tscrunch() / _config.cb_fscrunch() * _config.nbeams());
    if(nsamples % SKYWEAVER_CB_NSAMPLES_PER_BLOCK != 0){
        throw std::runtime_error("Number of samples must be a multiple of SKYWEAVER_CB_NSAMPLES_PER_BLOCK");
    }
    std::size_t nsamples_out = nsamples / _config.cb_tscrunch();
    if(nsamples_out % SKYWEAVER_CB_NSAMPLES_PER_HEAP != 0){
        throw std::runtime_error("Number of samples must be a multiple of SKYWEAVER_CB_NSAMPLES_PER_HEAP");
    }
    BOOST_LOG_TRIVIAL(debug) << "Resizing output buffer from " << output.size()
                             << " to " << output_size << " elements";
    output.resize({nsamples / _config.cb_tscrunch(),
                   _config.nchans() / _config.cb_fscrunch(),
                   _config.nbeams()});
    output.metalike(input);
    output.tsamp(input.tsamp() * _config.cb_tscrunch());
    std::size_t expected_weights_size =
        _config.nbeams() * _config.nantennas() * _config.nchans();
    if(weights.size() != expected_weights_size) {
        throw std::runtime_error("Unexpected size of weights vector");
    }
    dim3 grid(nsamples /
                  (SKYWEAVER_CB_NWARPS_PER_BLOCK * _config.cb_tscrunch()),
              _config.nchans() / _config.cb_fscrunch(),
              _config.nbeams() / SKYWEAVER_CB_WARP_SIZE);
    char2 const* ftpa_voltages_ptr = thrust::raw_pointer_cast(input.data());
    char2 const* fbpa_weights_ptr  = thrust::raw_pointer_cast(weights.data());
    typename BfTraits::QuantisedPowerType* tfb_powers_ptr =
        thrust::raw_pointer_cast(output.data());
    float const* power_scaling_ptr =
        thrust::raw_pointer_cast(output_scale.data());
    float const* power_offset_ptr =
        thrust::raw_pointer_cast(output_offset.data());
    typename BfTraits::RawPowerType const* ib_powers_ptr =
        thrust::raw_pointer_cast(ib_powers.data());
    int const* beamset_mapping_ptr =
        thrust::raw_pointer_cast(beamset_mapping.data());
    BOOST_LOG_TRIVIAL(debug) << "Executing beamforming kernel";
    kernels::bf_ftpa_general_k<BfTraits>
        <<<grid, SKYWEAVER_CB_NTHREADS, 0, stream>>>(
            (int2 const*)ftpa_voltages_ptr,
            (int2 const*)fbpa_weights_ptr,
            tfb_powers_ptr,
            power_scaling_ptr,
            power_offset_ptr,
            beamset_mapping_ptr,
            ib_powers_ptr,
            static_cast<int>(nsamples));
    CUDA_ERROR_CHECK(cudaStreamSynchronize(stream));
    BOOST_LOG_TRIVIAL(debug) << "Beamforming kernel complete";
}

template class CoherentBeamformer<
    SingleStokesBeamformerTraits<StokesParameter::I>>;
template class CoherentBeamformer<
    SingleStokesBeamformerTraits<StokesParameter::Q>>;
template class CoherentBeamformer<
    SingleStokesBeamformerTraits<StokesParameter::U>>;
template class CoherentBeamformer<
    SingleStokesBeamformerTraits<StokesParameter::V>>;
template class CoherentBeamformer<
    StokesTraits<StokesParameter::Q, StokesParameter::U>>;
template class CoherentBeamformer<
    StokesTraits<StokesParameter::I, StokesParameter::V>>;
template class CoherentBeamformer<FullStokesBeamformerTraits>;

} // namespace skyweaver
