#ifndef SKYWEAVER_TEST_TEST_UTILS_CUH
#define SKYWEAVER_TEST_TEST_UTILS_CUH

#include <cmath>
#include <complex>
#include <random>

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

std::string make_temporary_dir()
{
    char template_dirname[] = "/tmp/skyweaver_test_XXXXXX";
    char* directory_path    = mkdtemp(template_dirname);
    return std::string(directory_path);
}


} // namespace test
} // namespace skyweaver

#endif // SKYWEAVER_TEST_TEST_UTILS_CUH