#ifndef SKYWEAVER_TEST_BEAMFORMERPIPELINETESTER_CUH
#define SKYWEAVER_TEST_BEAMFORMERPIPELINETESTER_CUH

#include "skyweaver/BeamformerPipeline.cuh"

#include <gtest/gtest.h>
#include <vector>

namespace skyweaver
{
namespace test
{

class BeamformerPipelineTester: public ::testing::Test
{
  protected:
    void SetUp() override;
    void TearDown() override;

  public:
    BeamformerPipelineTester();
    ~BeamformerPipelineTester();

};

} // namespace test
} // namespace skyweaver

#endif // SKYWEAVER_TEST_BEAMFORMERPIPELINETESTER_CUH
