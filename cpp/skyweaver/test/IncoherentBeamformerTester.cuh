#ifndef SKYWEAVER_INCOHERENTBEAMFORMERTESTER_CUH
#define SKYWEAVER_INCOHERENTBEAMFORMERTESTER_CUH

#include "skyweaver/IncoherentBeamformer.cuh"
#include "skyweaver/PipelineConfig.hpp"
#include "thrust/host_vector.h"

#include <gtest/gtest.h>

namespace skyweaver
{
namespace test
{

template <typename BfTraits>
class IncoherentBeamformerTester: public ::testing::Test
{
  public:
    using BfTraitsType = BfTraits;
    typedef IncoherentBeamformer<BfTraits> IncoherentBeamformer;
    typedef IncoherentBeamformer::VoltageVectorType DeviceVoltageVectorType;
    typedef thrust::host_vector<typename DeviceVoltageVectorType::value_type>
        HostVoltageVectorType;
    typedef IncoherentBeamformer::PowerVectorType DevicePowerVectorType;
    typedef thrust::host_vector<typename DevicePowerVectorType::value_type>
        HostPowerVectorType;
    typedef IncoherentBeamformer::RawPowerVectorType DeviceRawPowerVectorType;
    typedef thrust::host_vector<typename DeviceRawPowerVectorType::value_type>
        HostRawPowerVectorType;
    typedef IncoherentBeamformer::ScalingVectorType DeviceScalingVectorType;
    typedef thrust::host_vector<typename DeviceScalingVectorType::value_type>
        HostScalingVectorType;

  protected:
    void SetUp() override;
    void TearDown() override;

  public:
    IncoherentBeamformerTester();
    ~IncoherentBeamformerTester();

  protected:
    void beamformer_c_reference(HostVoltageVectorType const& ftpa_voltages,
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
                                int nbeamsets);

    void compare_against_host(DeviceVoltageVectorType const& ftpa_voltages_gpu,
                              DeviceRawPowerVectorType& tf_powers_raw_gpu,
                              DevicePowerVectorType& tf_powers_gpu,
                              DeviceScalingVectorType const& scaling_vector,
                              DeviceScalingVectorType const& offset_vector,
                              DeviceScalingVectorType const& beamset_weights,
                              int ntimestamps,
                              int nbeamsets);

  protected:
    PipelineConfig _config;
    cudaStream_t _stream;
};

} // namespace test
} // namespace skyweaver

#endif // SKYWEAVER_INCOHERENTBEAMFORMERTESTER_CUH
