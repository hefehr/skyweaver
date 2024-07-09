#ifndef SKYWEAVER_TEST_COHERENTBEAMFORMERTESTER_CUH
#define SKYWEAVER_TEST_COHERENTBEAMFORMERTESTER_CUH

#include "skyweaver/CoherentBeamformer.cuh"
#include "skyweaver/IncoherentBeamformer.cuh"
#include "skyweaver/PipelineConfig.hpp"
#include "thrust/host_vector.h"

#include <gtest/gtest.h>

namespace skyweaver
{
namespace test
{

template <typename BfTraits>
class CoherentBeamformerTester: public ::testing::Test
{
  public:
    using BfTraitsType = BfTraits;
    typedef CoherentBeamformer<BfTraits> CoherentBeamformer;
    typedef IncoherentBeamformer<BfTraits> IncoherentBeamformer;
    typedef CoherentBeamformer::VoltageVectorType DeviceVoltageVectorType;
    typedef FTPAVoltagesH<typename DeviceVoltageVectorType::value_type>
        HostVoltageVectorType;
    typedef CoherentBeamformer::PowerVectorType DevicePowerVectorType;
    typedef TFBPowersH<typename DevicePowerVectorType::value_type>
        HostPowerVectorType;
    typedef IncoherentBeamformer::PowerVectorType DeviceIBPowerVectorType;
    typedef BTFPowersH<typename DeviceIBPowerVectorType::value_type>
        HostIBPowerVectorType;
    typedef IncoherentBeamformer::RawPowerVectorType DeviceRawIBPowerVectorType;
    typedef BTFPowersH<typename DeviceRawIBPowerVectorType::value_type>
        HostRawIBPowerVectorType;
    typedef CoherentBeamformer::WeightsVectorType DeviceWeightsVectorType;
    typedef thrust::host_vector<typename DeviceWeightsVectorType::value_type>
        HostWeightsVectorType;
    typedef CoherentBeamformer::ScalingVectorType DeviceScalingVectorType;
    typedef thrust::host_vector<typename DeviceScalingVectorType::value_type>
        HostScalingVectorType;
    typedef CoherentBeamformer::MappingVectorType DeviceMappingVectorType;
    typedef thrust::host_vector<typename DeviceMappingVectorType::value_type>
        HostMappingVectorType;

  protected:
    void SetUp() override;
    void TearDown() override;

  public:
    CoherentBeamformerTester();
    ~CoherentBeamformerTester();

  protected:
    void beamformer_c_reference(HostVoltageVectorType const& ftpa_voltages,
                                HostWeightsVectorType const& fbpa_weights,
                                HostPowerVectorType& btf_powers,
                                int nchannels,
                                int tscrunch,
                                int fscrunch,
                                int nsamples,
                                int nbeams,
                                int nantennas,
                                int npol,
                                float const* scales,
                                float const* offsets,
                                float const* antenna_weights,
                                int const* beamset_mapping);

    void compare_against_host(DeviceVoltageVectorType const& ftpa_voltages_gpu,
                              DeviceWeightsVectorType const& fbpa_weights_gpu,
                              DeviceScalingVectorType const& scales_gpu,
                              DeviceScalingVectorType const& offsets_gpu,
                              DeviceScalingVectorType const& antenna_weights,
                              DeviceMappingVectorType const& beamset_mapping,
                              DevicePowerVectorType& btf_powers_gpu,
                              int nsamples);

  protected:
    PipelineConfig _config;
    cudaStream_t _stream;
};

} // namespace test
} // namespace skyweaver

#endif // SKYWEAVER_TEST_COHERENTBEAMFORMERTESTER_CUH
