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
    typedef CoherentBeamformer::VoltageVectorTypeD VoltageVectorTypeD;
    typedef FTPAVoltagesH<typename VoltageVectorTypeD::value_type>
    VoltageVectorTypeH;
    typedef CoherentBeamformer::PowerVectorTypeD PowerVectorTypeD;
    typedef TFBPowersH<typename PowerVectorTypeD::value_type>
        HostPowerVectorType;
    typedef IncoherentBeamformer::PowerVectorTypeD IBPowerVectorTypeD;
    typedef BTFPowersH<typename IBPowerVectorTypeD::value_type>
        HostIBPowerVectorType;
    typedef IncoherentBeamformer::RawPowerVectorTypeD RawIBPowerVectorTypeD;
    typedef BTFPowersH<typename RawIBPowerVectorTypeD::value_type>
        HostRawIBPowerVectorType;
    typedef CoherentBeamformer::WeightsVectorTypeD WeightsVectorTypeD;
    typedef thrust::host_vector<typename WeightsVectorTypeD::value_type>
        WeightsVectorTypeH;
    typedef CoherentBeamformer::ScalingVectorTypeD ScalingVectorTypeD;
    typedef thrust::host_vector<typename ScalingVectorTypeD::value_type>
        HostScalingVectorType;
    typedef CoherentBeamformer::MappingVectorTypeD MappingVectorTypeD;
    typedef thrust::host_vector<typename MappingVectorTypeD::value_type>
        MappingVectorTypeH;

  protected:
    void SetUp() override;
    void TearDown() override;

  public:
    CoherentBeamformerTester();
    ~CoherentBeamformerTester();

  protected:
    void beamformer_c_reference(VoltageVectorTypeH const& ftpa_voltages,
                                WeightsVectorTypeH const& fbpa_weights,
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

    void compare_against_host(VoltageVectorTypeD const& ftpa_voltages_gpu,
                              WeightsVectorTypeD const& fbpa_weights_gpu,
                              ScalingVectorTypeD const& scales_gpu,
                              ScalingVectorTypeD const& offsets_gpu,
                              ScalingVectorTypeD const& antenna_weights,
                              MappingVectorTypeD const& beamset_mapping,
                              PowerVectorTypeD& btf_powers_gpu,
                              int nsamples);

  protected:
    PipelineConfig _config;
    cudaStream_t _stream;
};

} // namespace test
} // namespace skyweaver

#endif // SKYWEAVER_TEST_COHERENTBEAMFORMERTESTER_CUH
