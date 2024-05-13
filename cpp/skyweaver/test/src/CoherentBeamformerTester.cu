#include "psrdada_cpp/common.hpp"
#include "psrdada_cpp/cuda_utils.hpp"
#include "skyweaver/IncoherentBeamformer.cuh"
#include "skyweaver/beamformer_utils.cuh"
#include "skyweaver/skyweaver_constants.hpp"
#include "skyweaver/test/CoherentBeamformerTester.cuh"

#include <cmath>
#include <complex>
#include <random>
#include <fstream>

namespace skyweaver
{
namespace test
{

template <typename VectorType>
void dump_host_vector(VectorType const& vec, std::string filename)
{
    std::ofstream infile;
    infile.open(filename.c_str(), std::ifstream::out | std::ifstream::binary);
    if(!infile.is_open()) {
        throw std::runtime_error("Unable to open file");
    }
    infile.write(reinterpret_cast<char const*>(thrust::raw_pointer_cast(vec.data())),
                 vec.size() * sizeof(typename VectorType::value_type));
    infile.close();
}

template <typename VectorType>
static void dump_device_vector(VectorType const& vec, std::string filename)
{
    thrust::host_vector<typename VectorType::value_type> host_vec = vec;
    dump_host_vector(host_vec, filename);
}

CoherentBeamformerTester::CoherentBeamformerTester()
    : ::testing::Test(), _stream(0)
{
}

CoherentBeamformerTester::~CoherentBeamformerTester()
{
}

void CoherentBeamformerTester::SetUp()
{
    CUDA_ERROR_CHECK(cudaStreamCreate(&_stream));
}

void CoherentBeamformerTester::TearDown()
{
    CUDA_ERROR_CHECK(cudaStreamDestroy(_stream));
}

void CoherentBeamformerTester::beamformer_c_reference(
    HostVoltageVectorType const& ftpa_voltages,
    HostWeightsVectorType const& fbpa_weights,
    HostPowerVectorType& tbtf_powers,
    int nchannels,
    int tscrunch,
    int fscrunch,
    int nsamples,
    int nbeams,
    int nantennas,
    int npol,
    float const* scales,
    float const* offsets)
{
    double power_sum    = 0.0;
    double power_sq_sum = 0.0;
    double ib_power_sum = 0.0;
    std::size_t count   = 0;
    for(int channel_idx = 0; channel_idx < nchannels; channel_idx += fscrunch) {
        BOOST_LOG_TRIVIAL(debug)
            << "Beamformer C reference: "
            << static_cast<int>(100.0f * (channel_idx + 1.0f) / nchannels)
            << "% complete";
        for(int sample_idx = 0; sample_idx < nsamples; sample_idx += tscrunch) {
            for(int beam_idx = 0; beam_idx < nbeams; ++beam_idx) {
                float power    = 0.0f;
                float ib_power = 0.0f;
                for(int sub_channel_idx = channel_idx;
                    sub_channel_idx < channel_idx + fscrunch;
                    ++sub_channel_idx) {
                    for(int sample_offset = 0; sample_offset < tscrunch;
                        ++sample_offset) {
                        cuFloatComplex p0_accumulator = {0, 0};
                        cuFloatComplex p1_accumulator = {0, 0};
                        // new loop
                        for(int antenna_idx = 0; antenna_idx < nantennas;
                            ++antenna_idx) {
                            const int ftpa_voltages_partial_idx =
                                nantennas * npol * nsamples * sub_channel_idx +
                                nantennas * npol *
                                    (sample_idx + sample_offset) +
                                +antenna_idx;
                            const int p0_idx = ftpa_voltages_partial_idx;
                            const int p1_idx =
                                ftpa_voltages_partial_idx + nantennas;
                            char2 ant_p0_c2       = ftpa_voltages[p0_idx];
                            char2 ant_p1_c2       = ftpa_voltages[p1_idx];
                            cuFloatComplex ant_p0 = make_cuFloatComplex(
                                static_cast<float>(ant_p0_c2.x),
                                static_cast<float>(ant_p0_c2.y));
                            cuFloatComplex ant_p1 = make_cuFloatComplex(
                                static_cast<float>(ant_p1_c2.x),
                                static_cast<float>(ant_p1_c2.y));
                            ib_power += calculate_stokes(ant_p0, ant_p1);
                            int fbpa_weights_idx =
                                nantennas * nbeams * sub_channel_idx +
                                nantennas * beam_idx + antenna_idx;
                            char2 weight_c2 = fbpa_weights[fbpa_weights_idx];
                            cuFloatComplex weight = make_cuFloatComplex(
                                static_cast<float>(weight_c2.x),
                                static_cast<float>(weight_c2.y));
                            cuFloatComplex p0 = cuCmulf(weight, ant_p0);
                            cuFloatComplex p1 = cuCmulf(weight, ant_p1);
                            p0_accumulator    = cuCaddf(p0_accumulator, p0);
                            p1_accumulator    = cuCaddf(p1_accumulator, p1);
                        }
                        power +=
                            calculate_stokes(p0_accumulator, p1_accumulator);
                        // end new loop
                    }
                }
                const int output_sample_idx = sample_idx / tscrunch;
                const int nsamps_out        = nsamples / tscrunch;
                const int output_chan_idx   = channel_idx / fscrunch;
                const int nchans_out        = nchannels / fscrunch;
                const int tf_size           = nsamps_out * nchans_out;
                int output_idx = beam_idx * tf_size + output_sample_idx * nchans_out + output_chan_idx;
                power_sum += power;
                ib_power_sum += ib_power;
                power_sq_sum += power * power;
                ++count;
#if SKYWEAVER_IB_SUBTRACTION
                float powerf32 = ((power - (127.0f * 127.0f * ib_power)) /
                                  scales[output_chan_idx]);
#else
                float powerf32 = ((power - offsets[output_chan_idx]) /
                                  scales[output_chan_idx]);
#endif // SKYWEAVER_IB_SUBTRACTION
                tbtf_powers[output_idx] =
                    (int8_t)fmaxf(-127.0f, fminf(127.0f, powerf32));
            }
        }
    }
    double power_mean = power_sum / count;
    BOOST_LOG_TRIVIAL(debug) << "Average power level: " << power_mean;
    BOOST_LOG_TRIVIAL(debug)
        << "Power variance: " << power_sq_sum / count - power_mean * power_mean;
}

void CoherentBeamformerTester::compare_against_host(
    DeviceVoltageVectorType const& ftpa_voltages_gpu,
    DeviceWeightsVectorType const& fbpa_weights_gpu,
    DeviceScalingVectorType const& scales_gpu,
    DeviceScalingVectorType const& offsets_gpu,
    DevicePowerVectorType& btf_powers_gpu,
    int nsamples)
{
    HostVoltageVectorType ftpa_voltages_host = ftpa_voltages_gpu;
    HostWeightsVectorType fbpa_weights_host  = fbpa_weights_gpu;
    HostPowerVectorType btf_powers_cuda      = btf_powers_gpu;
    HostPowerVectorType btf_powers_host(btf_powers_gpu.size());

    HostScalingVectorType scales  = scales_gpu;
    HostScalingVectorType offsets = offsets_gpu;

    beamformer_c_reference(ftpa_voltages_host,
                           fbpa_weights_host,
                           btf_powers_host,
                           _config.nchans(),
                           _config.cb_tscrunch(),
                           _config.cb_fscrunch(),
                           nsamples,
                           _config.nbeams(),
                           _config.nantennas(),
                           _config.npol(),
                           thrust::raw_pointer_cast(scales.data()),
                           thrust::raw_pointer_cast(offsets.data()));
    for(int ii = 0; ii < btf_powers_host.size(); ++ii) {
        EXPECT_NEAR(btf_powers_host[ii], btf_powers_cuda[ii], 1);
    }
}

TEST_F(CoherentBeamformerTester, representative_noise_test)
{
#if SKYWEAVER_IB_SUBTRACTION
    BOOST_LOG_TRIVIAL(info) << "Running with IB subtraction";
#else
    BOOST_LOG_TRIVIAL(info) << "Running without IB subtraction";
#endif
    const float input_level = 32.0f;
    const double pi         = std::acos(-1);
    _config.output_level(input_level);

    float scale =
        std::pow(127.0f * input_level *
                     std::sqrt(static_cast<float>(_config.nantennas())),
                 2);
    float dof =
        2 * _config.cb_tscrunch() * _config.cb_fscrunch() * _config.npol();
    float offset_val = (scale * dof);
    float scale_val  = (scale * std::sqrt(2 * dof) / _config.output_level());
    DeviceScalingVectorType cb_scales(_config.nchans() / _config.cb_fscrunch(),
                                      scale_val);
    DeviceScalingVectorType cb_offsets(_config.nchans() / _config.cb_fscrunch(),
                                       offset_val);
    BOOST_LOG_TRIVIAL(info) << "CB scaling: " << scale_val;
    BOOST_LOG_TRIVIAL(info) << "CB offset: " << offset_val;

    std::default_random_engine generator;
    std::normal_distribution<float> normal_dist(0.0, input_level);
    std::uniform_real_distribution<float> uniform_dist(0.0, 2 * pi);

    CoherentBeamformer coherent_beamformer(_config);
    IncoherentBeamformer incoherent_beamformer(_config);

    std::size_t ntimestamps = max(
        1L,
        SKYWEAVER_CB_PACKET_SIZE / (_config.nchans() / _config.cb_fscrunch()) /
            (_config.nsamples_per_heap() / _config.cb_tscrunch()));
    ntimestamps =
        max(ntimestamps,
            SKYWEAVER_CB_NSAMPLES_PER_BLOCK / _config.nsamples_per_heap());
    std::size_t input_size =
        (ntimestamps * _config.nantennas() * _config.nchans() *
         _config.nsamples_per_heap() * _config.npol());
    BOOST_LOG_TRIVIAL(info) << "FTPA input dims: " << _config.nchans() << ", " << ntimestamps * _config.nsamples_per_heap() << ", " << _config.npol() << ", " << _config.nantennas();
    int nsamples = _config.nsamples_per_heap() * ntimestamps;

    std::size_t weights_size =
        _config.nantennas() * _config.nchans() * _config.nbeams();

    HostVoltageVectorType ftpa_voltages_host(input_size);
    for(int ii = 0; ii < ftpa_voltages_host.size(); ++ii) {
        ftpa_voltages_host[ii].x =
            static_cast<int8_t>(std::lround(normal_dist(generator)));
        ftpa_voltages_host[ii].y =
            static_cast<int8_t>(std::lround(normal_dist(generator)));
    }

    HostWeightsVectorType fbpa_weights_host(weights_size);
    for(int ii = 0; ii < fbpa_weights_host.size(); ++ii) {
        // Build complex weight as C * exp(i * theta).
        std::complex<double> val =
            127.0f *
            std::exp(std::complex<float>(0.0f, uniform_dist(generator)));
        fbpa_weights_host[ii].x = static_cast<int8_t>(std::lround(val.real()));
        fbpa_weights_host[ii].y = static_cast<int8_t>(std::lround(val.imag()));
    }

    DeviceVoltageVectorType ftpa_voltages_gpu = ftpa_voltages_host;
    DeviceWeightsVectorType fbpa_weights_gpu  = fbpa_weights_host;
    DevicePowerVectorType btf_powers_gpu;

    // Note that below even though this is for the IB we have to use the
    // CB scrunching parameters to make sure we get the right data out.
    float ib_scale = std::pow(input_level, 2);
    float ib_dof   = 2 * _config.cb_tscrunch() * _config.cb_fscrunch() *
                   _config.nantennas() * _config.npol();
    float ib_power_offset = ib_scale * ib_dof;
    float ib_power_scaling =
        ib_scale * std::sqrt(2 * ib_dof) / _config.output_level();
    DeviceScalingVectorType ib_scales(_config.nchans() / _config.cb_fscrunch(),
                                      ib_power_scaling);
    DeviceScalingVectorType ib_offset(_config.nchans() / _config.cb_fscrunch(),
                                      ib_power_offset);
    DevicePowerVectorType tf_powers_gpu;
    DeviceRawPowerVectorType tf_powers_raw_gpu;

    dump_device_vector(ftpa_voltages_gpu, "ftpa_voltages_gpu.bin");
    incoherent_beamformer.beamform(ftpa_voltages_gpu,
                                   tf_powers_raw_gpu,
                                   tf_powers_gpu,
                                   ib_scales,
                                   ib_offset,
                                   _stream);
    dump_device_vector(tf_powers_raw_gpu, "tf_powers_raw_gpu.bin");
    dump_device_vector(fbpa_weights_gpu, "fbpa_weights_gpu.bin");
    coherent_beamformer.beamform(ftpa_voltages_gpu,
                                 fbpa_weights_gpu,
                                 cb_scales,
                                 cb_offsets,
                                 tf_powers_raw_gpu,
                                 btf_powers_gpu,
                                 _stream);
    dump_device_vector(btf_powers_gpu, "btf_powers_gpu.bin");
    compare_against_host(ftpa_voltages_gpu,
                         fbpa_weights_gpu,
                         cb_scales,
                         cb_offsets,
                         btf_powers_gpu,
                         nsamples);
}

} // namespace test
} // namespace skyweaver
