#ifndef SKYWEAVER_TEST_OBSERVATIONHEADERTESTER_HPP
#define SKYWEAVER_TEST_OBSERVATIONHEADERTESTER_HPP

#include "skyweaver/ObservationHeader.hpp"

#include <gtest/gtest.h>
#include <vector>

namespace skyweaver
{
namespace test
{

class ObservationHeaderTester: public ::testing::Test
{
  protected:
    void SetUp() override;
    void TearDown() override;

  public:
    ObservationHeaderTester();
    ~ObservationHeaderTester();
};

} // namespace test
} // namespace skyweaver

#endif // SKYWEAVER_TEST_OBSERVATIONHEADERTESTER_HPP
