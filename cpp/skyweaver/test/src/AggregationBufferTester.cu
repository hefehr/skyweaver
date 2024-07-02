#include "skyweaver/test/AggregationBufferTester.cuh"
#include "skyweaver/AggregationBuffer.cuh"
#include "skyweaver/types.cuh"
#include "skyweaver/test/test_utils.cuh"
#include "thrust/device_vector.h"
#include <vector>

namespace skyweaver {
namespace test {

typedef ::testing::Types<
    std::vector<int8_t>, std::vector<float>, std::vector<char4>, std::vector<float4>,
    thrust::host_vector<int8_t>, thrust::host_vector<float>, thrust::host_vector<char4>, thrust::host_vector<float4>,
    thrust::device_vector<int8_t>, thrust::device_vector<float>, thrust::device_vector<char4>, thrust::device_vector<float4>
    > TestTypes;

TYPED_TEST_SUITE(AggregationBufferTester, TestTypes);

TYPED_TEST(AggregationBufferTester, PushBackTest) {
    using ContainerType = TypeParam;
    typedef typename ContainerType::value_type T;

    AggregationBuffer<T> buffer(
        std::bind(&AggregationBufferTester<ContainerType>::MockCallback, this, std::placeholders::_1), 10, 0, 1);
    ContainerType data(10, T{});
    buffer.push_back(data);
    EXPECT_EQ(this->callback_count, 1);
    EXPECT_EQ(this->callback_arg, data);
}

TYPED_TEST(AggregationBufferTester, PushBackMultiTest) {
    using ContainerType = TypeParam;
    typedef typename ContainerType::value_type T;

    AggregationBuffer<T> buffer(
        std::bind(&AggregationBufferTester<ContainerType>::MockCallback, this, std::placeholders::_1), 10, 0, 1);

    ContainerType data(100, T{});
    sequence(data);

    buffer.push_back(data);

    ContainerType expected_data(10);
    std::copy(data.end() - 10, data.end(), expected_data.begin());
    EXPECT_EQ(this->callback_count, 10);
    EXPECT_EQ(this->callback_arg, expected_data);
}

TYPED_TEST(AggregationBufferTester, PushBackOverlappedTest) {
    using ContainerType = TypeParam;
    typedef typename ContainerType::value_type T;
    AggregationBuffer<T> buffer(
        std::bind(&AggregationBufferTester<ContainerType>::MockCallback, this, std::placeholders::_1), 10, 1, 2);
    ContainerType data(100, T{});
    sequence(data);
    buffer.push_back(data);
    ContainerType expected_data((10 + 1) * 2);
    std::size_t expected_callbacks = 4;
    std::copy(data.begin() + 2 * 10 * (expected_callbacks-1), data.begin() + 2 * 10 * (expected_callbacks) + 2, expected_data.begin());
    EXPECT_EQ(this->callback_count, expected_callbacks);
    EXPECT_EQ(this->callback_arg, expected_data);
}

TYPED_TEST(AggregationBufferTester, InvalidLengthTest) {
    using ContainerType = TypeParam;
    typedef typename ContainerType::value_type T;
    AggregationBuffer<T> buffer(
        std::bind(&AggregationBufferTester<ContainerType>::MockCallback, this, std::placeholders::_1), 10, 1, 2);
    ContainerType data(101, T{});
    EXPECT_THROW(buffer.push_back(data), std::runtime_error);
}


} // namespace test
} // namespace skyweaver