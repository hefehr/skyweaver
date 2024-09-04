#ifndef SKYWEAVER_TEST_BUFFERED_DISPENSER_TESTER_HPP
#define SKYWEAVER_TEST_BUFFERED_DISPENSER_TESTER_HPP
#include "skyweaver/PipelineConfig.hpp"
#include "skyweaver/CoherentDedisperser.cuh"
#include <fstream>
#include <gtest/gtest.h>
#include <memory>

namespace skyweaver
{
namespace test
{

class BufferedDispenserTester: public ::testing::Test
{
  protected:
    PipelineConfig pipeline_config;
    CoherentDedisperserConfig dedisp_config;
    std::size_t nantennas;
    std::size_t nchans;
    std::size_t npols;
    std::size_t nbits;
    std::size_t max_delay_samps;

  protected:
    void SetUp() override;
    void TearDown() override;

  public:
    BufferedDispenserTester();
    ~BufferedDispenserTester();
};

} // namespace test
} // namespace skyweaver

#endif // SKYWEAVER_TEST_BUFFERED_DISPENSER_TESTER_HPP