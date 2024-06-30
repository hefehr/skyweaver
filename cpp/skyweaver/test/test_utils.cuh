#ifndef SKYWEAVER_TEST_TEST_UTILS_CUH
#define SKYWEAVER_TEST_TEST_UTILS_CUH

#include "skyweaver/types.cuh"
#include <thrust/transform.h>
#include <thrust/execution_policy.h>
#include <cmath>
#include <complex>
#include <gtest/gtest.h>
#include <iostream>
#include <random>
#include <type_traits>

namespace skyweaver
{
namespace test
{
namespace
{
template <typename T, typename DistributionType, typename EngineType>
T generate_sample(DistributionType& dist, EngineType& engine)
{
    if constexpr(std::is_integral<T>::value) {
        return static_cast<T>(std::lround(dist(engine)));
    } else {
        return static_cast<T>(dist(engine));
    }
}

template <typename T, typename DistributionType, typename EngineType>
T generate_complex_sample(DistributionType& dist, EngineType& engine)
{
    T out;
    out.x =
        generate_sample<decltype(out.x), DistributionType, EngineType>(dist,
                                                                       engine);
    out.y =
        generate_sample<decltype(out.y), DistributionType, EngineType>(dist,
                                                                       engine);
    return out;
}

} // namespace

template <typename VectorType>
void random_normal(VectorType& vec, std::size_t n, float mean, float std)
{
    std::default_random_engine generator;
    std::normal_distribution<float> normal_dist(mean, std);
    vec.resize(n);
    for(int ii = 0; ii < vec.size(); ++ii) {
        vec[ii] = generate_sample<typename VectorType::value_type,
                                  decltype(normal_dist),
                                  decltype(generator)>(normal_dist, generator);
    }
}

template <typename VectorType>
void random_normal_complex(VectorType& vec,
                           std::size_t n,
                           float mean,
                           float std)
{
    std::default_random_engine generator;
    std::normal_distribution<float> normal_dist(mean, std);
    vec.resize(n);
    for(int ii = 0; ii < vec.size(); ++ii) {
        vec[ii] = generate_complex_sample<typename VectorType::value_type,
                                          decltype(normal_dist),
                                          decltype(generator)>(normal_dist,
                                                               generator);
    }
}

template <typename T, typename X>
typename std::enable_if<std::is_arithmetic_v<T>, void>::type
expect_near(T const& a, T const& b, X const& c)
{
    EXPECT_NEAR(a, b, c);
}

template <typename T, typename X>
typename std::enable_if<is_vec4_v<T>, void>::type
expect_near(T const& a, T const& b, X const& c)
{
    EXPECT_NEAR(a.x, b.x, c);
    EXPECT_NEAR(a.y, b.y, c);
    EXPECT_NEAR(a.z, b.z, c);
    EXPECT_NEAR(a.w, b.w, c);
}

template <typename T, typename X>
typename std::enable_if<std::is_arithmetic_v<T>, void>::type
expect_relatively_near(T const& a, T const& b, X const& c)
{
    EXPECT_NEAR(a, b, std::abs(a * c));
}

template <typename T, typename X>
typename std::enable_if<is_vec4_v<T>, void>::type
expect_relatively_near(T const& a, T const& b, X const& c)
{
    EXPECT_NEAR(a.x, b.x, std::abs(a.x * c));
    EXPECT_NEAR(a.y, b.y, std::abs(a.y * c));
    EXPECT_NEAR(a.z, b.z, std::abs(a.z * c));
    EXPECT_NEAR(a.w, b.w, std::abs(a.w * c));
}

template <typename T>
struct Sequence {
    __host__ __device__
    T operator()(const int& i) const {
        return value_traits<T>::zero() + value_traits<T>::one() * i;
    }
};

template <typename VectorType>
void sequence(VectorType& vec){
    typedef typename VectorType::value_type T;
    thrust::transform(
        thrust::counting_iterator<int>(0),
        thrust::counting_iterator<int>(vec.size()),
        vec.begin(),
        Sequence<T>()
    );
}

} // namespace test
} // namespace skyweaver

#endif // SKYWEAVER_TEST_TEST_UTILS_CUH