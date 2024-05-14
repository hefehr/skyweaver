#ifndef SKYWEAVER_WEIGHTSMANAGERTESTER_CUH
#define SKYWEAVER_WEIGHTSMANAGERTESTER_CUH

#include "skyweaver/PipelineConfig.hpp"
#include "skyweaver/WeightsManager.cuh"

#include <gtest/gtest.h>

namespace skyweaver
{
namespace test
{

class WeightsManagerTester: public ::testing::Test
{
  public:
    typedef WeightsManager::DelayVectorType DelayVectorType;
    typedef WeightsManager::WeightsVectorType WeightsVectorType;
    typedef WeightsManager::TimeType TimeType;

  protected:
    void SetUp() override;
    void TearDown() override;

  public:
    WeightsManagerTester();
    ~WeightsManagerTester();

  protected:
    void
    calc_weights_c_reference(thrust::host_vector<float3> const& delay_models,
                             thrust::host_vector<char2>& weights,
                             std::vector<double> const& channel_frequencies,
                             int nantennas,
                             int nbeams,
                             int nchans,
                             double current_epoch,
                             double delay_epoch,
                             double tstep,
                             int ntsteps);

    void compare_against_host(DelayVectorType const& delays,
                              WeightsVectorType const& weights,
                              TimeType current_epoch,
                              TimeType delay_epoch);

  protected:
    PipelineConfig _config;
    cudaStream_t _stream;
};

} // namespace test
} // namespace skyweaver

#endif // SKYWEAVER_WEIGHTSMANAGERTESTER_CUH
