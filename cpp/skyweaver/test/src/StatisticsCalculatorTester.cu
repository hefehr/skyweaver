
#include "psrdada_cpp/cuda_utils.hpp"
#include "skyweaver/DescribedVector.hpp"
#include "skyweaver/PipelineConfig.hpp"
#include "skyweaver/skyweaver_constants.hpp"
#include "skyweaver/test/StatisticsCalculatorTester.cuh"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/host_vector.h>
#include <thrust/random.h>
#include <thrust/random/normal_distribution.h>
#include <vector>

#define EXPECT_RELATIVE_ERROR()

namespace skyweaver
{
namespace test
{

// Borrowed from https://www.johndcook.com/blog/skewness_kurtosis/
class RunningStats
{
  public:
    RunningStats();
    void clear();
    void push(double x);
    long long num_data_values() const;
    double mean() const;
    double variance() const;
    double standard_deviation() const;
    double skewness() const;
    double kurtosis() const;

  private:
    long long n;
    double M1, M2, M3, M4;
};

RunningStats::RunningStats()
{
    clear();
}

void RunningStats::clear()
{
    n  = 0;
    M1 = M2 = M3 = M4 = 0.0;
}

void RunningStats::push(double x)
{
    double delta, delta_n, delta_n2, term1;

    long long n1 = n;
    n++;
    delta    = x - M1;
    delta_n  = delta / n;
    delta_n2 = delta_n * delta_n;
    term1    = delta * delta_n * n1;
    M1 += delta_n;
    M4 += term1 * delta_n2 * (n * n - 3 * n + 3) + 6 * delta_n2 * M2 -
          4 * delta_n * M3;
    M3 += term1 * delta_n * (n - 2) - 3 * delta_n * M2;
    M2 += term1;
}

long long RunningStats::num_data_values() const
{
    return n;
}

double RunningStats::mean() const
{
    return M1;
}

double RunningStats::variance() const
{
    return M2 / (n - 1.0);
}

double RunningStats::standard_deviation() const
{
    return sqrt(variance());
}

double RunningStats::skewness() const
{
    return sqrt(double(n)) * M3 / pow(M2, 1.5);
}

double RunningStats::kurtosis() const
{
    return double(n) * M4 / (M2 * M2) - 3.0;
}

StatisticsCalculatorTester::StatisticsCalculatorTester()
    : ::testing::Test(), _stream(0)
{
}

StatisticsCalculatorTester::~StatisticsCalculatorTester()
{
}

void StatisticsCalculatorTester::SetUp()
{
    CUDA_ERROR_CHECK(cudaStreamCreate(&_stream));
}

void StatisticsCalculatorTester::TearDown()
{
    CUDA_ERROR_CHECK(cudaStreamDestroy(_stream));
}

void StatisticsCalculatorTester::compare_against_host(
    FTPAVoltagesH<char2>& data,
    FPAStatsH<Statistics>& gpu_results) const
{
    std::size_t fpa_size =
        _config.nantennas() * _config.npol() * _config.nchans();

    // Create accumulators for each of the output samples
    // We calculate the stats this way so that the behaviour matches
    // what is expected from the GPU implementation.
    std::vector<RunningStats> stats(fpa_size);

    // Data is in FTPA order
    std::size_t nsamples = data.size() / (fpa_size);

    std::size_t input_idx = 0;
    for(std::size_t f = 0; f < _config.nchans(); ++f) {
        for(std::size_t t = 0; t < nsamples; ++t) {
            for(std::size_t p = 0; p < _config.npol(); ++p) {
                for(std::size_t a = 0; a < _config.nantennas(); ++a) {
                    std::size_t output_idx =
                        (f * _config.nantennas() * _config.npol()) +
                        (p * _config.nantennas()) + a;
                    char2 d = data[input_idx];
                    stats[output_idx].push(double(d.x));
                    stats[output_idx].push(double(d.y));
                    ++input_idx;
                }
            }
        }
    }

    for(std::size_t stats_idx = 0; stats_idx < fpa_size; ++stats_idx) {
        // Even though the values are doubles, we do not expect the host
        // and device codes to actually return identical values given
        // differences in operation order. Hence we compare as if the
        // values are floats.
        EXPECT_FLOAT_EQ(gpu_results[stats_idx].mean, stats[stats_idx].mean());
        EXPECT_FLOAT_EQ(gpu_results[stats_idx].std,
                        stats[stats_idx].standard_deviation());
        EXPECT_FLOAT_EQ(gpu_results[stats_idx].skew,
                        stats[stats_idx].skewness());
        EXPECT_FLOAT_EQ(gpu_results[stats_idx].kurtosis,
                        stats[stats_idx].kurtosis());
    }
}

TEST_F(StatisticsCalculatorTester, test_normal_dist)
{
    // Make some input data
    std::size_t nsamples = 1024;
    thrust::default_random_engine rng(1337);
    thrust::random::normal_distribution<float> dist(0.0f, 20.0f);
    FTPAVoltagesH<char2> ftpa_voltages_h(
        {_config.nchans(), nsamples, _config.npol(), _config.nantennas()});
    thrust::generate(ftpa_voltages_h.begin(), ftpa_voltages_h.end(), [&] {
        char2 val;
        val.x = static_cast<int8_t>(std::clamp(dist(rng), -127.0f, 127.0f));
        val.y = static_cast<int8_t>(std::clamp(dist(rng), -127.0f, 127.0f));
        return val;
    });
    FTPAVoltagesD<char2> ftpa_voltages = ftpa_voltages_h;
    StatisticsCalculator calculator(_config, _stream);
    calculator.calculate_statistics(ftpa_voltages);
    FPAStatsD<Statistics> stats = calculator.statistics();
    FPAStatsH<Statistics> stats_h;
    stats_h.like(stats);
    stats_h = stats;
    compare_against_host(ftpa_voltages_h, stats_h);
}

TEST_F(StatisticsCalculatorTester, test_file_writer)
{
    // Make some input data
    std::size_t nsamples = 1024;
    FTPAVoltagesH<char2> ftpa_voltages_h(
        {_config.nchans(), nsamples, _config.npol(), _config.nantennas()});
    FTPAVoltagesD<char2> ftpa_voltages = ftpa_voltages_h;
    StatisticsCalculator calculator(_config, _stream);
    calculator.calculate_statistics(ftpa_voltages);
    FPAStatsD<Statistics> stats = calculator.statistics();
}

} // namespace test
} // namespace skyweaver
