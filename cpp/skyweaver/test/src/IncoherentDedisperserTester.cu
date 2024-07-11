#include "skyweaver/test/IncoherentDedisperserTester.cuh"
#include "skyweaver/DescribedVector.hpp"
#include "skyweaver/AggregationBuffer.cuh"
#include "skyweaver/types.cuh"
#include "skyweaver/test/test_utils.cuh"
#include "thrust/host_vector.h"

namespace skyweaver
{
namespace test
{

template <typename Traits>
IncoherentDedisperserTester<Traits>::IncoherentDedisperserTester()
    : ::testing::Test()
{
}

template <typename Traits>
IncoherentDedisperserTester<Traits>::~IncoherentDedisperserTester()
{
}

template <typename Traits>
void IncoherentDedisperserTester<Traits>::SetUp()
{
}

template <typename Traits>
void IncoherentDedisperserTester<Traits>::TearDown()
{
}

typedef ::testing::Types<IDTesterTraits<int8_t, int8_t>,
                         IDTesterTraits<char4, char4>>
        TestTypes;
TYPED_TEST_SUITE(IncoherentDedisperserTester, TestTypes);

TYPED_TEST(IncoherentDedisperserTester, zero_dm_delays_test)
{
    using Traits = TypeParam;
    IncoherentDedisperser dedisperser(this->_config, 0.0f);
    auto const& delays = dedisperser.delays();
    ASSERT_EQ(delays.size(), 1 * this->_config.channel_frequencies().size()) << "Delay vector has unexpected length";
    for (int const& delay: delays) {    
        EXPECT_EQ(delay, 0) << "Delays not equal to zero";
    }
    ASSERT_EQ(dedisperser.max_delay(), 0);
}

TYPED_TEST(IncoherentDedisperserTester, ones_test)
{
    using Traits = TypeParam;
    this->_config.bandwidth(32e6);
    this->_config.centre_frequency(580e6);
    std::vector<float> dms = {0.0f, 10.0f, 20.0f, 30.0f, 40.0f};
    IncoherentDedisperser dedisperser(this->_config, dms);
    auto const& delays = dedisperser.delays();
    ASSERT_EQ(delays.size(), dms.size() * this->_config.channel_frequencies().size()) << "Delay vector has unexpected length";
    ASSERT_GT(dedisperser.max_delay(), 0);
    std::size_t nsamples = dedisperser.max_delay() * 2;
    thrust::host_vector<typename Traits::InputType> data(
        this->_config.nbeams() * nsamples * this->_config.nchans(), 
        value_traits<typename Traits::InputType>::one());
    TDBPowersH<typename Traits::OutputType> output;
    dedisperser.dedisperse(data, output);
    ASSERT_EQ(output.size(), (nsamples - dedisperser.max_delay()) * dms.size() * this->_config.nbeams());
    for (auto const& val: output)
    {
        EXPECT_EQ(val, value_traits<typename Traits::OutputType>::one() * std::sqrt(this->_config.nchans()));
    }
}

TYPED_TEST(IncoherentDedisperserTester, too_few_samples_test)
{
    using Traits = TypeParam;
    this->_config.bandwidth(32e6);
    this->_config.centre_frequency(580e6);
    std::vector<float> dms = {0.0f, 10.0f, 20.0f, 30.0f, 40.0f};
    IncoherentDedisperser dedisperser(this->_config, dms);
    thrust::host_vector<typename Traits::InputType> data(
        this->_config.nbeams() * dedisperser.max_delay() * this->_config.nchans());
    TDBPowersH<typename Traits::OutputType> output;
    EXPECT_THROW(dedisperser.dedisperse(data, output), std::runtime_error);
}

TYPED_TEST(IncoherentDedisperserTester, ones_test_wtscrunch)
{
    using Traits = TypeParam;
    this->_config.bandwidth(32e6);
    this->_config.centre_frequency(580e6);
    std::size_t tscrunch = 2;
    std::vector<float> dms = {0.0f, 10.0f, 20.0f, 30.0f, 40.0f};
    IncoherentDedisperser dedisperser(this->_config, dms, tscrunch);
    auto const& delays = dedisperser.delays();
    ASSERT_EQ(delays.size(), dms.size() * this->_config.channel_frequencies().size()) << "Delay vector has unexpected length";
    ASSERT_GT(dedisperser.max_delay(), 0);
    std::size_t nsamples = dedisperser.max_delay() * 2 + dedisperser.max_delay();
    thrust::host_vector<typename Traits::InputType> data(
        this->_config.nbeams() * nsamples * this->_config.nchans(), 
        value_traits<typename Traits::InputType>::one());
    TDBPowersH<typename Traits::OutputType> output;
    dedisperser.dedisperse(data, output);
    ASSERT_EQ(output.size(), (nsamples - dedisperser.max_delay())/tscrunch * dms.size() * this->_config.nbeams());
    for (auto const& val: output)
    {
        EXPECT_EQ(val, static_cast<typename Traits::OutputType>(value_traits<typename Traits::OutputType>::one() * std::sqrt(this->_config.nchans() * tscrunch)));
    }
}


} // namespace test
} // namespace skyweaver