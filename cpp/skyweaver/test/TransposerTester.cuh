#ifndef SKYWEAVER_TEST_TRANSPOSERTESTER_CUH
#define SKYWEAVER_TEST_TRANSPOSERTESTER_CUH

#include "skyweaver/PipelineConfig.hpp"
#include "skyweaver/Transposer.cuh"
#include "thrust/host_vector.h"
#include <gtest/gtest.h>

namespace skyweaver {
namespace test {

class TransposerTester: public ::testing::TestWithParam<std::size_t>
{
public:
    typedef Transposer::VoltageType DeviceVoltageType;
    typedef thrust::host_vector<char2> HostVoltageType;


protected:
    void SetUp() override;
    void TearDown() override;

public:
    TransposerTester();
    ~TransposerTester();

protected:
    void transpose_c_reference(
        HostVoltageType const& input,
        HostVoltageType& output,
        int input_nantennas,
        int output_nantennas,
        int nchans,
        int ntimestamps);

    void compare_against_host(
        DeviceVoltageType const& gpu_input,
        DeviceVoltageType const& gpu_output,
        std::size_t input_nantennas,
        std::size_t ntimestamps);

protected:
    PipelineConfig _config;
    cudaStream_t _stream;
};

} //namespace test
} //namespace skyweaver

#endif //SKYWEAVER_TEST_TRANSPOSERTESTER_CUH
