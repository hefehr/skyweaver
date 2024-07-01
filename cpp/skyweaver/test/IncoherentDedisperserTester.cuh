#ifndef SKYWEAVER_INCOHERENTDEDISPERSERTESTER_CUH
#define SKYWEAVER_INCOHERENTDEDISPERSERTESTER_CUH

#include "skyweaver/IncoherentDedisperser.cuh"
#include "skyweaver/PipelineConfig.hpp"
#include <gtest/gtest.h>

namespace skyweaver
{
namespace test
{

template <typename InputType_, typename OutputType_>
struct IDTesterTraits
{
    typedef InputType_ InputType;
    typedef OutputType_ OutputType;
};

template <class Traits>
class IncoherentDedisperserTester: public ::testing::Test
{
  protected:
    void SetUp() override;
    void TearDown() override;

  public:
    IncoherentDedisperserTester();
    ~IncoherentDedisperserTester();

  protected:
    PipelineConfig _config;
};

} // namespace test
} // namespace skyweaver

#endif //SKYWEAVER_INCOHERENTDEDISPERSERTESTER_CUH