#include "psrdada_cpp/common.hpp"
#include "psrdada_cpp/cuda_utils.hpp"
#include "skyweaver/beamformer_utils.cuh"
#include "skyweaver/skyweaver_constants.hpp"
#include "skyweaver/test/IncoherentBeamformerTester.cuh"

#include <cmath>
#include <complex>
#include <random>

namespace skyweaver
{
namespace test
{

IncoherentBeamformerTester::IncoherentBeamformerTester()
    : ::testing::Test(), _stream(0)
{
}

IncoherentBeamformerTester::~IncoherentBeamformerTester()
{
}

void IncoherentBeamformerTester::SetUp()
{
    CUDA_ERROR_CHECK(cudaStreamCreate(&_stream));
}

void IncoherentBeamformerTester::TearDown()
{
    CUDA_ERROR_CHECK(cudaStreamDestroy(_stream));
}

void IncoherentBeamformerTester::beamformer_c_reference(
    HostVoltageVectorType const& ftpa_voltages,
    HostRawPowerVectorType& tf_powers_raw,
    HostPowerVectorType& tf_powers,
    int nchannels,
    int tscrunch,
    int fscrunch,
    int ntimestamps,
    int nantennas,
    int nsamples_per_timestamp,
    HostScalingVectorType const& scale,
    HostScalingVectorType const& offset)
{
    static_assert(SKYWEAVER_NPOL == 2, "Tests only work for dual poln data.");
    const int nchans_out = SKYWEAVER_NCHANS / SKYWEAVER_IB_FSCRUNCH;
    const int a          = SKYWEAVER_NANTENNAS;
    const int pa         = SKYWEAVER_NPOL * a;
    const int tpa        = ntimestamps * pa;

    for(int F_idx = 0; F_idx < SKYWEAVER_NCHANS;
        F_idx += SKYWEAVER_IB_FSCRUNCH) {
        for(int T_idx = 0; T_idx < SKYWEAVER_NCHANS;
            T_idx += SKYWEAVER_IB_TSCRUNCH) {
            float power = 0.0f;
            for(int f_idx = F_idx; f_idx < F_idx + SKYWEAVER_IB_FSCRUNCH;
                ++f_idx) {
                for(int t_idx = T_idx; t_idx < T_idx + SKYWEAVER_IB_TSCRUNCH;
                    ++t_idx) {
                    for(int a_idx = 0; a_idx < SKYWEAVER_NANTENNAS; ++a_idx) {
                        int input_p0_idx = f_idx * tpa + t_idx * pa + a_idx;
                        int input_p1_idx = f_idx * tpa + t_idx * pa + a + a_idx;
                        char2 p0_v       = ftpa_voltages[input_p0_idx];
                        char2 p1_v       = ftpa_voltages[input_p1_idx];
                        cuFloatComplex p0 =
                            make_cuFloatComplex((float)p0_v.x, (float)p0_v.y);
                        cuFloatComplex p1 =
                            make_cuFloatComplex((float)p1_v.x, (float)p1_v.y);
                        power += calculate_stokes(p0, p1);
                    }
                }
            }
            int subband_idx           = F_idx / SKYWEAVER_IB_FSCRUNCH;
            int subbint_idx           = T_idx / SKYWEAVER_IB_TSCRUNCH;
            int output_idx            = subbint_idx * nchans_out + subband_idx;
            tf_powers_raw[output_idx] = power;
            float scaled_power =
                ((power - offset[subband_idx]) / scale[subband_idx]);
            tf_powers[output_idx] =
                (int8_t)fmaxf(-127.0f, fminf(127.0f, scaled_power));
        }
    }
}

void IncoherentBeamformerTester::compare_against_host(
    DeviceVoltageVectorType const& ftpa_voltages_gpu,
    DeviceRawPowerVectorType& tf_powers_raw_gpu,
    DevicePowerVectorType& tf_powers_gpu,
    DeviceScalingVectorType const& scaling_vector,
    DeviceScalingVectorType const& offset_vector,
    int ntimestamps)
{
    HostVoltageVectorType ftpa_voltages_host  = ftpa_voltages_gpu;
    HostPowerVectorType tf_powers_cuda        = tf_powers_gpu;
    HostRawPowerVectorType tf_powers_raw_cuda = tf_powers_raw_gpu;
    HostScalingVectorType h_scaling_vector    = scaling_vector;
    HostScalingVectorType h_offset_vector     = offset_vector;
    HostRawPowerVectorType tf_powers_raw_host(tf_powers_raw_gpu.size());
    HostPowerVectorType tf_powers_host(tf_powers_gpu.size());
    beamformer_c_reference(ftpa_voltages_host,
                           tf_powers_raw_host,
                           tf_powers_host,
                           _config.nchans(),
                           _config.ib_tscrunch(),
                           _config.ib_fscrunch(),
                           ntimestamps,
                           _config.nantennas(),
                           _config.nsamples_per_heap(),
                           h_scaling_vector,
                           h_offset_vector);
    for(int ii = 0; ii < tf_powers_host.size(); ++ii) {
        EXPECT_TRUE(std::abs(static_cast<int>(tf_powers_host[ii]) -
                             tf_powers_cuda[ii]) <= 1);
        EXPECT_TRUE(
            std::fabs((tf_powers_raw_host[ii] - tf_powers_raw_cuda[ii]) /
                      tf_powers_raw_host[ii]) <= 1e-5);
    }
}

TEST_F(IncoherentBeamformerTester, ib_representative_noise_test)
{
    float input_level = 32.0f;
    _config.output_level(32.0f);
    std::default_random_engine generator;
    std::normal_distribution<float> normal_dist(0.0, 32.0f);
    IncoherentBeamformer incoherent_beamformer(_config);
    std::size_t ntimestamps = 32;
    std::size_t input_size =
        (ntimestamps * _config.nantennas() * _config.nchans() *
         _config.nsamples_per_heap() * _config.npol());
    HostVoltageVectorType ftpa_voltages_host(input_size);
    for(int ii = 0; ii < ftpa_voltages_host.size(); ++ii) {
        ftpa_voltages_host[ii].x =
            static_cast<int8_t>(std::lround(normal_dist(generator)));
        ftpa_voltages_host[ii].y =
            static_cast<int8_t>(std::lround(normal_dist(generator)));
    }

    float ib_scale = std::pow(input_level, 2);
    float ib_dof   = 2 * _config.ib_tscrunch() * _config.ib_fscrunch() *
                   _config.nantennas() * _config.npol();
    float ib_power_offset = ib_scale * ib_dof;
    float ib_power_scaling =
        ib_scale * std::sqrt(2 * ib_dof) / _config.output_level();
    DeviceScalingVectorType scales(_config.nchans() / _config.ib_fscrunch(),
                                   ib_power_scaling);
    DeviceScalingVectorType offset(_config.nchans() / _config.ib_fscrunch(),
                                   ib_power_offset);
    DeviceVoltageVectorType ftpa_voltages_gpu = ftpa_voltages_host;
    DevicePowerVectorType tf_powers_gpu;
    DeviceRawPowerVectorType tf_powers_raw_gpu;
    incoherent_beamformer.beamform(ftpa_voltages_gpu,
                                   tf_powers_raw_gpu,
                                   tf_powers_gpu,
                                   scales,
                                   offset,
                                   _stream);
    compare_against_host(ftpa_voltages_gpu,
                         tf_powers_raw_gpu,
                         tf_powers_gpu,
                         scales,
                         offset,
                         ntimestamps);
}

} // namespace test
} // namespace skyweaver
