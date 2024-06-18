#ifndef SKYWEAVER_TEST_BUFFERED_DISPENSER_TESTER_HPP
#define SKYWEAVER_TEST_BUFFERED_DISPENSER_TESTER_HPP
#include "skyweaver/MultiFileReader.cuh"
#include "skyweaver/ObservationHeader.hpp"
#include "skyweaver/PipelineConfig.hpp"

#include <fstream>
#include <gtest/gtest.h>
#include <memory>
#include <thrust/host_vector.h>

namespace skyweaver
{
namespace test
{

class BufferedDispenserTester: public ::testing::Test
{
  protected:
    PipelineConfig pipeline_config;
    std::unique_ptr<MultiFileReader> multi_file_reader;
    std::size_t nantennas;
    std::size_t nchans;
    std::size_t npols;
    std::size_t nbits;

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