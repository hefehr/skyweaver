#ifndef SKYWEAVER_TEST_COHERENTDEDISPERSERTESTER_HPP
#define SKYWEAVER_TEST_COHERENTDEDISPERSERTESTER_HPP


#include <gtest/gtest.h>
#include "skyweaver/CoherentDedisperser.cuh"
#include "skyweaver/CoherentDedisperser.cuh"
#include "skyweaver/PipelineConfig.hpp"
#include <vector>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <algorithm>

namespace skyweaver
{
namespace test
{

class CoherentDedisperserTester: public ::testing::Test
{
  protected:
    CoherentDedisperserConfig dedisp_config;
    std::vector<float> dms;
    std::size_t nantennas;
    std::size_t nchans;
    std::size_t npols;
    std::size_t nbits;    
    std::size_t max_delay_samps;
    std::size_t fft_length;


    void SetUp() override;
    void TearDown() override;

  public:
    CoherentDedisperserTester();
    ~CoherentDedisperserTester();

};

} // namespace test
} // namespace skyweaver

#endif // SKYWEAVER_TEST_COHERENTDEDISPERSERTESTER_HPP