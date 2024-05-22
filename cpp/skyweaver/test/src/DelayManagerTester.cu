#include "psrdada_cpp/cuda_utils.hpp"
#include "skyweaver/test/DelayManagerTester.cuh"
#include "thrust/host_vector.h"

#include <cstdlib>
#include <iostream>

namespace skyweaver
{
namespace test
{

DelayManagerTester::DelayManagerTester(): ::testing::Test(), _stream(0)
{
}

DelayManagerTester::~DelayManagerTester()
{
}

void DelayManagerTester::SetUp()
{
    _config.delay_file("data/test_delays.bin");
    CUDA_ERROR_CHECK(cudaStreamCreate(&_stream));
    if (_config.nantennas() < 57)
    {
        GTEST_SKIP();
    }
    if (_config.nbeams() < 67)
    {
        GTEST_SKIP();
    }
}

void DelayManagerTester::TearDown()
{
    CUDA_ERROR_CHECK(cudaStreamDestroy(_stream));
}

/**
 * The data/ directory contains a test delay file called
 * test_delays.bin.
 *
 * It contains 5 delay models for 67 beams and 57 antennas.
 *
 * The first valid epoch is   1708082168.957
 * The last valid epoch is 1708082468.957
 *
 * delay offsets are in the range 1e-7 to 1e-10
 * delay rates are in the range 1e-12 to 1e-14
 * weights are either 1 or 0
 */

TEST_F(DelayManagerTester, test_valid_read_first_block)
{
    DelayManager delay_manager(_config, _stream);
    DelayManager::DelayVectorDType const& delays = delay_manager.delays(1708082169.0);
    ASSERT_EQ(delays.size(), _config.nbeams() * _config.nantennas());
}

TEST_F(DelayManagerTester, test_valid_read_nth_block)
{
    DelayManager delay_manager(_config, _stream);
    DelayManager::DelayVectorDType const& delays = delay_manager.delays(1708082409.957);
    ASSERT_EQ(delays.size(), _config.nbeams() * _config.nantennas());
}


TEST_F(DelayManagerTester, test_too_early_epoch)
{
    DelayManager delay_manager(_config, _stream);
    //Test an epoch that is before the start of the validity window
    EXPECT_THROW(
        DelayManager::DelayVectorDType const& delays = delay_manager.delays(1708082165.0),
        InvalidDelayEpoch
    );
}

TEST_F(DelayManagerTester, test_too_late_epoch)
{
    DelayManager delay_manager(_config, _stream);
    //Test an epoch that is after the end of the validity window
    EXPECT_THROW(
        DelayManager::DelayVectorDType const& delays = delay_manager.delays(1708082470.957),
        std::runtime_error
    );
}

} // namespace test
} // namespace skyweaver