#ifndef SKYWEAVER_TEST_STATISTICSCALCULATORTESTER_CUH
#define SKYWEAVER_TEST_STATISTICSCALCULATORTESTER_CUH

#include "skyweaver/PipelineConfig.hpp"
#include "skyweaver/StatisticsCalculator.cuh"
#include "thrust/host_vector.h"

#include <gtest/gtest.h>
#include <string>

namespace skyweaver
{
namespace test
{
class StatisticsCalculatorTester: public ::testing::Test
{
  protected:
    void SetUp() override;
    void TearDown() override;

  public:
    StatisticsCalculatorTester();
    ~StatisticsCalculatorTester();
    void
    compare_against_host(FTPAVoltagesH<char2>& data,
                         FPAStatsH<Statistics>& gpu_results) const;     

  protected:
    cudaStream_t _stream;
    PipelineConfig _config;
};

} // namespace test
} // namespace skyweaver

#endif // SKYWEAVER_TEST_STATISTICSCALCULATORTESTER_CUH