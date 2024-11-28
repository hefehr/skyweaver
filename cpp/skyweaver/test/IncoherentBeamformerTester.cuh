#ifndef SKYWEAVER_INCOHERENTBEAMFORMERTESTER_CUH
#define SKYWEAVER_INCOHERENTBEAMFORMERTESTER_CUH

#include "skyweaver/DescribedVector.hpp"
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
    typedef IncoherentBeamformer::VoltageVectorTypeD VoltageVectorTypeD;
    typedef FTPAVoltagesH<typename VoltageVectorTypeD::value_type>
        VoltageVectorTypeH;
    typedef IncoherentBeamformer::PowerVectorTypeD DevicePowerVectorType;
    typedef BTFPowersH<typename DevicePowerVectorType::value_type>
        HostPowerVectorType;
    typedef IncoherentBeamformer::RawPowerVectorTypeD DeviceRawPowerVectorType;
    typedef BTFPowersH<typename DeviceRawPowerVectorType::value_type>
        HostRawPowerVectorType;
    typedef IncoherentBeamformer::ScalingVectorTypeD ScalingVectorTypeD;
    typedef thrust::host_vector<typename ScalingVectorTypeD::value_type>
        HostScalingVectorType;
    typedef IncoherentBeamformer::BeamsetWeightsVectorTypeD BeamsetWeightsVectorTypeD;
    typedef thrust::host_vector<typename BeamsetWeightsVectorTypeD::value_type>
        HostBeamsetWeightsVectorType;

  protected:
    void SetUp() override;
    void TearDown() override;

  public:
    IncoherentBeamformerTester();
    ~IncoherentBeamformerTester();

  protected:
    void beamformer_c_reference(VoltageVectorTypeH const& ftpa_voltages,
                                HostRawPowerVectorType& tf_powers_raw,
                                HostPowerVectorType& tf_powers,
                                int nchannels,
                                int tscrunch,
                                int fscrunch,
                                int ntimestamps,
                                int nantennas,
                                HostScalingVectorType const& scale,
                                HostScalingVectorType const& offset,
                                HostBeamsetWeightsVectorType const& beamset_weights,
                                int nbeamsets);

    void compare_against_host(VoltageVectorTypeD const& ftpa_voltages_gpu,
                              DeviceRawPowerVectorType& tf_powers_raw_gpu,
                              DevicePowerVectorType& tf_powers_gpu,
                              ScalingVectorTypeD const& scaling_vector,
                              ScalingVectorTypeD const& offset_vector,
                              BeamsetWeightsVectorTypeD const& beamset_weights,
                              int ntimestamps,
                              int nbeamsets);

  protected:
    PipelineConfig _config;
    cudaStream_t _stream;
};

} // namespace test
} // namespace skyweaver

#endif // SKYWEAVER_INCOHERENTBEAMFORMERTESTER_CUH
