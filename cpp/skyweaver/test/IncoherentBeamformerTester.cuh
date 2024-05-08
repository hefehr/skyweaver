#ifndef SKYWEAVER_INCOHERENTBEAMFORMERTESTER_CUH
#define SKYWEAVER_INCOHERENTBEAMFORMERTESTER_CUH

#include "skyweaver/PipelineConfig.hpp"
#include "skyweaver/IncoherentBeamformer.cuh"
#include "thrust/host_vector.h"
#include <gtest/gtest.h>

namespace skyweaver {
namespace test {

class IncoherentBeamformerTester: public ::testing::Test
{
public:
    typedef IncoherentBeamformer::VoltageVectorType DeviceVoltageVectorType;
    typedef thrust::host_vector<char2> HostVoltageVectorType;
    typedef IncoherentBeamformer::PowerVectorType DevicePowerVectorType;
    typedef thrust::host_vector<int8_t> HostPowerVectorType;
    typedef IncoherentBeamformer::RawPowerVectorType DeviceRawPowerVectorType;
    typedef thrust::host_vector<float> HostRawPowerVectorType;
    typedef IncoherentBeamformer::ScalingVectorType DeviceScalingVectorType;
    typedef thrust::host_vector<float> HostScalingVectorType;

protected:
    void SetUp() override;
    void TearDown() override;

public:
    IncoherentBeamformerTester();
    ~IncoherentBeamformerTester();

protected:
    void beamformer_c_reference(
        HostVoltageVectorType const& taftp_voltages,
        HostRawPowerVectorType& tf_powers_raw,
        HostPowerVectorType& tf_powers,
        int nchannels,
        int tscrunch,
        int fscrunch,
        int ntimestamps,
        int nantennas,
        int nsamples_per_timestamp,
        HostScalingVectorType const& scale,
        HostScalingVectorType const& offset);

    void compare_against_host(
        DeviceVoltageVectorType const& taftp_voltages_gpu,
        DeviceRawPowerVectorType& tf_powers_raw_gpu,
        DevicePowerVectorType& tf_powers_gpu,
        DeviceScalingVectorType const& scaling_vector,
        DeviceScalingVectorType const& offset_vector,
        int ntimestamps);

protected:
    PipelineConfig _config;
    cudaStream_t _stream;
};

} //namespace test
} //namespace skyweaver

#endif //SKYWEAVER_INCOHERENTBEAMFORMERTESTER_CUH
