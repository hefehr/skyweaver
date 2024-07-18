#ifndef SKYWEAVER_TEST_MULTIFILEWRITER_TESTER_CUH
#define SKYWEAVER_TEST_MULTIFILEWRITER_TESTER_CUH
#include "skyweaver/MultiFileWriter.cuh"
#include "skyweaver/ObservationHeader.hpp"
#include "skyweaver/PipelineConfig.hpp"

#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <memory>
#include <thrust/host_vector.h>

namespace skyweaver
{
namespace test
{

class MultiFileWriterTester: public ::testing::Test
{
  public:
    MultiFileWriterTester();
    ~MultiFileWriterTester();

  protected:
    void SetUp() override;
    void TearDown() override;

    PipelineConfig _config;
};

} // namespace test
} // namespace skyweaver

#endif // SKYWEAVER_TEST_MULTIFILEWRITER_TESTER_CUH