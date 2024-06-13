#ifndef SKYWEAVER_TEST_COHERENTBEAMFORMERTESTER_CUH
#define SKYWEAVER_TEST_COHERENTBEAMFORMERTESTER_CUH

#include "skyweaver/PipelineConfig.hpp"
#include "skyweaver/CoherentBeamformer.cuh"
#include "thrust/host_vector.h"
#include <gtest/gtest.h>

namespace skyweaver {
namespace test {

class CoherentBeamformerTester: public ::testing::Test
{
public:
    typedef CoherentBeamformer::VoltageVectorType DeviceVoltageVectorType;
    typedef thrust::host_vector<char2> HostVoltageVectorType;
    typedef CoherentBeamformer::PowerVectorType DevicePowerVectorType;
    typedef thrust::host_vector<char> HostPowerVectorType;
    typedef CoherentBeamformer::RawPowerVectorType DeviceRawPowerVectorType;
    typedef thrust::host_vector<float> HostRawPowerVectorType;
    typedef CoherentBeamformer::WeightsVectorType DeviceWeightsVectorType;
    typedef thrust::host_vector<char2> HostWeightsVectorType;
    typedef CoherentBeamformer::ScalingVectorType DeviceScalingVectorType;
    typedef thrust::host_vector<float> HostScalingVectorType;
    typedef CoherentBeamformer::MappingVectorType DeviceMappingVectorType;
    typedef thrust::host_vector<int> HostMappingVectorType;

protected:
    void SetUp() override;
    void TearDown() override;

public:
    CoherentBeamformerTester();
    ~CoherentBeamformerTester();

protected:
    void beamformer_c_reference(
        HostVoltageVectorType const& ftpa_voltages,
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

    void compare_against_host(
        DeviceVoltageVectorType const& ftpa_voltages_gpu,
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

} //namespace test
} //namespace skyweaver

#endif //SKYWEAVER_TEST_COHERENTBEAMFORMERTESTER_CUH
