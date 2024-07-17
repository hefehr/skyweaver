#include "psrdada_cpp/common.hpp"
#include "psrdada_cpp/cuda_utils.hpp"
#include "skyweaver/beamformer_utils.cuh"
#include "skyweaver/skyweaver_constants.hpp"
#include "skyweaver/test/IncoherentBeamformerTester.cuh"
#include "skyweaver/test/test_utils.cuh"

#include <cmath>
#include <complex>
#include <random>

namespace skyweaver
{
namespace test
{

template <typename BfTraits>
IncoherentBeamformerTester<BfTraits>::IncoherentBeamformerTester()
    : ::testing::Test(), _stream(0)
{
}

template <typename BfTraits>
IncoherentBeamformerTester<BfTraits>::~IncoherentBeamformerTester()
{
}

template <typename BfTraits>
void IncoherentBeamformerTester<BfTraits>::SetUp()
{
    CUDA_ERROR_CHECK(cudaStreamCreate(&_stream));
}

template <typename BfTraits>
void IncoherentBeamformerTester<BfTraits>::TearDown()
{
    CUDA_ERROR_CHECK(cudaStreamDestroy(_stream));
}

template <typename BfTraits>
void IncoherentBeamformerTester<BfTraits>::beamformer_c_reference(
    VoltageVectorTypeH const& ftpa_voltages,
    HostRawPowerVectorType& tf_powers_raw,
    HostPowerVectorType& tf_powers,
    int nchannels,
    int tscrunch,
    int fscrunch,
    int ntimestamps,
    int nantennas,
    HostScalingVectorType const& scale,
    HostScalingVectorType const& offset,
    HostScalingVectorType const& beamset_weights,
    int nbeamsets)
{
    static_assert(SKYWEAVER_NPOL == 2, "Tests only work for dual poln data.");
    const int nchans_out   = nchannels / fscrunch;
    const int nsamples_out = ntimestamps / tscrunch;
    const int a            = nantennas;
    const int pa           = SKYWEAVER_NPOL * a;
    const int tpa          = ntimestamps * pa;
    for(int beamset_idx = 0; beamset_idx < nbeamsets; ++beamset_idx) {
        for(int F_idx = 0; F_idx < nchannels; F_idx += fscrunch) {
            for(int T_idx = 0; T_idx < ntimestamps; T_idx += tscrunch) {
                typename BfTraits::RawPowerType power = BfTraits::zero_power;
                for(int f_idx = F_idx; f_idx < F_idx + fscrunch; ++f_idx) {
                    for(int t_idx = T_idx; t_idx < T_idx + tscrunch; ++t_idx) {
                        for(int a_idx = 0; a_idx < nantennas; ++a_idx) {
                            float weight =
                                beamset_weights[beamset_idx * nantennas +
                                                a_idx];
                            int input_p0_idx = f_idx * tpa + t_idx * pa + a_idx;
                            int input_p1_idx =
                                f_idx * tpa + t_idx * pa + a + a_idx;
                            char2 p0_v = ftpa_voltages[input_p0_idx];
                            char2 p1_v = ftpa_voltages[input_p1_idx];
                            cuFloatComplex p0 =
                                make_cuFloatComplex((float)p0_v.x,
                                                    (float)p0_v.y);
                            cuFloatComplex p1 =
                                make_cuFloatComplex((float)p1_v.x,
                                                    (float)p1_v.y);
                            BfTraits::integrate_weighted_stokes(p0,
                                                                p1,
                                                                power,
                                                                weight);
                        }
                    }
                }
                int subband_idx = F_idx / fscrunch;
                int subbint_idx = T_idx / tscrunch;
                int output_idx  = beamset_idx * nsamples_out * nchans_out +
                                 subbint_idx * nchans_out + subband_idx;
                int scloff_idx = beamset_idx * nchans_out + subband_idx;
                tf_powers_raw[output_idx] = power;
                typename BfTraits::RawPowerType scaled_power =
                    BfTraits::rescale(power,
                                      offset[scloff_idx],
                                      scale[scloff_idx]);
                tf_powers[output_idx] = BfTraits::clamp(scaled_power);
            }
        }
    }
}

template <typename BfTraits>
void IncoherentBeamformerTester<BfTraits>::compare_against_host(
    DeviceVoltageVectorType const& ftpa_voltages_gpu,
    DeviceRawPowerVectorType& tf_powers_raw_gpu,
    DevicePowerVectorType& tf_powers_gpu,
    DeviceScalingVectorType const& scaling_vector,
    DeviceScalingVectorType const& offset_vector,
    DeviceScalingVectorType const& beamset_weights,
    int ntimestamps,
    int nbeamsets)
{
    VoltageVectorTypeH ftpa_voltages_host     = ftpa_voltages_gpu;
    HostPowerVectorType tf_powers_cuda        = tf_powers_gpu;
    HostRawPowerVectorType tf_powers_raw_cuda = tf_powers_raw_gpu;
    HostScalingVectorType h_scaling_vector    = scaling_vector;
    HostScalingVectorType h_offset_vector     = offset_vector;
    HostScalingVectorType h_beamset_weights   = beamset_weights;
    HostRawPowerVectorType tf_powers_raw_host;
    tf_powers_raw_host.like(tf_powers_raw_gpu);
    HostPowerVectorType tf_powers_host;
    tf_powers_host.like(tf_powers_gpu);
    beamformer_c_reference(ftpa_voltages_host,
                           tf_powers_raw_host,
                           tf_powers_host,
                           _config.nchans(),
                           _config.ib_tscrunch(),
                           _config.ib_fscrunch(),
                           ntimestamps,
                           _config.nantennas(),
                           h_scaling_vector,
                           h_offset_vector,
                           h_beamset_weights,
                           nbeamsets);
    for(int ii = 0; ii < tf_powers_host.size(); ++ii) {
        expect_near(tf_powers_host[ii], tf_powers_cuda[ii], 1);
        expect_relatively_near(tf_powers_raw_host[ii],
                               tf_powers_raw_cuda[ii],
                               1e-5);
    }
}

typedef ::testing::Types<SingleStokesBeamformerTraits<StokesParameter::I>,
                         SingleStokesBeamformerTraits<StokesParameter::Q>,
                         SingleStokesBeamformerTraits<StokesParameter::U>,
                         SingleStokesBeamformerTraits<StokesParameter::V>,
                         FullStokesBeamformerTraits>
    StokesTypes;
TYPED_TEST_SUITE(IncoherentBeamformerTester, StokesTypes);

TYPED_TEST(IncoherentBeamformerTester, ib_representative_noise_test)
{
    using BfTraits    = typename TestFixture::BfTraitsType;
    using IBT         = IncoherentBeamformerTester<BfTraits>;
    auto& config      = this->_config;
    float input_level = 32.0f;
    config.output_level(32.0f);
    std::default_random_engine generator;
    std::normal_distribution<float> normal_dist(0.0, 32.0f);
    IncoherentBeamformer<BfTraits> incoherent_beamformer(config);
    std::size_t ntimestamps = 8192;
    std::size_t input_size =
        (ntimestamps * config.nantennas() * config.nchans() * config.npol());
    typename IBT::VoltageVectorTypeH ftpa_voltages_host(
        {config.nchans(), ntimestamps, config.npol(), config.nantennas()});
    for(int ii = 0; ii < ftpa_voltages_host.size(); ++ii) {
        ftpa_voltages_host[ii].x =
            static_cast<int8_t>(std::lround(normal_dist(generator)));
        ftpa_voltages_host[ii].y =
            static_cast<int8_t>(std::lround(normal_dist(generator)));
    }

    float ib_scale = std::pow(input_level, 2);
    float ib_dof   = 2 * config.ib_tscrunch() * config.ib_fscrunch() *
                   config.nantennas() * config.npol();
    float ib_power_offset = ib_scale * ib_dof;
    float ib_power_scaling =
        ib_scale * std::sqrt(2 * ib_dof) / config.output_level();

    for(int nbeamsets = 1; nbeamsets < 5; ++nbeamsets) {
        typename IBT::DeviceScalingVectorType scales(
            config.nchans() / config.ib_fscrunch() * nbeamsets,
            ib_power_scaling);
        typename IBT::DeviceScalingVectorType offset(
            config.nchans() / config.ib_fscrunch() * nbeamsets,
            ib_power_offset);
        typename IBT::DeviceScalingVectorType beamset_weights(
            config.nantennas() * nbeamsets,
            1.0f);

        typename IBT::DeviceVoltageVectorType ftpa_voltages_gpu =
            ftpa_voltages_host;
        typename IBT::DevicePowerVectorType tf_powers_gpu;
        typename IBT::DeviceRawPowerVectorType tf_powers_raw_gpu;
        incoherent_beamformer.beamform(ftpa_voltages_gpu,
                                       tf_powers_raw_gpu,
                                       tf_powers_gpu,
                                       scales,
                                       offset,
                                       beamset_weights,
                                       nbeamsets,
                                       this->_stream);
        this->compare_against_host(ftpa_voltages_gpu,
                                   tf_powers_raw_gpu,
                                   tf_powers_gpu,
                                   scales,
                                   offset,
                                   beamset_weights,
                                   ntimestamps,
                                   nbeamsets);
    }
}

} // namespace test
} // namespace skyweaver
