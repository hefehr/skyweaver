#ifndef SKYWEAVER_TEST_AGGREGATIONBUFFERTESTER_CUH
#define SKYWEAVER_TEST_AGGREGATIONBUFFERTESTER_CUH

#include "skyweaver/AggregationBuffer.cuh"
#include "thrust/host_vector.h"

#include <gtest/gtest.h>

namespace skyweaver
{
namespace test
{

template <class Container>
class AggregationBufferTester: public ::testing::Test
{
  protected:
    typedef typename Container::value_type T;

    void SetUp() override
    {
        callback_count = 0;
        callback_arg.clear();
    }

    void TearDown() override {}

    int callback_count;
    thrust::host_vector<T> callback_arg;

  public:
    void MockCallback(thrust::host_vector<T> const& arg)
    {
        ++callback_count;
        callback_arg = arg;
    }
};

} // namespace test
} // namespace skyweaver

#endif // SKYWEAVER_TEST_AGGREGATIONBUFFERTESTER_CUH