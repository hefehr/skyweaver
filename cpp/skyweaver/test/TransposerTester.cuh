#ifndef SKYWEAVER_TEST_TRANSPOSERTESTER_CUH
#define SKYWEAVER_TEST_TRANSPOSERTESTER_CUH

#include "skyweaver/PipelineConfig.hpp"
#include "skyweaver/Transposer.cuh"
#include "thrust/host_vector.h"

#include <gtest/gtest.h>

namespace skyweaver
{
namespace test
{

struct TransposerParameters {
    std::size_t nantennas;
    std::size_t ntimestamps;
};

class TransposerTester: public ::testing::TestWithParam<TransposerParameters>
{
  public:
    typedef Transposer::InputVoltageTypeD DeviceInputVoltageTypeD;
    typedef Transposer::OutputVoltageTypeD DeviceOutputVoltageTypeD;
    typedef TAFTPVoltagesH<char2> HostInputVoltageTypeD;
    typedef FTPAVoltagesH<char2> HostOutputVoltageTypeD;

  protected:
    void SetUp() override;
    void TearDown() override;

  public:
    TransposerTester();
    ~TransposerTester();

  protected:
    void transpose_c_reference(HostInputVoltageTypeD const& input,
                               HostOutputVoltageTypeD& output,
                               std::size_t input_nantennas,
                               std::size_t output_nantennas,
                               std::size_t nchans,
                               std::size_t ntimestamps);

    void compare_against_host(DeviceInputVoltageTypeD const& gpu_input,
                              DeviceOutputVoltageTypeD const& gpu_output,
                              std::size_t input_nantennas,
                              std::size_t ntimestamps);

  protected:
    PipelineConfig _config;
    cudaStream_t _stream;
};

} // namespace test
} // namespace skyweaver

#endif // SKYWEAVER_TEST_TRANSPOSERTESTER_CUH
