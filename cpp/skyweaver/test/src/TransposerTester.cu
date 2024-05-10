#include "psrdada_cpp/cuda_utils.hpp"
#include "skyweaver/skyweaver_constants.hpp"
#include "skyweaver/test/TransposerTester.cuh"

namespace skyweaver
{
namespace test
{

TransposerTester::TransposerTester(): ::testing::TestWithParam<std::size_t>(), _stream(0)
{
}

TransposerTester::~TransposerTester()
{
}

void TransposerTester::SetUp()
{
    CUDA_ERROR_CHECK(cudaStreamCreate(&_stream));
}

void TransposerTester::TearDown()
{
    CUDA_ERROR_CHECK(cudaStreamDestroy(_stream));
}

void TransposerTester::transpose_c_reference(HostVoltageType const& input,
                                             HostVoltageType& output,
                                             int input_nantennas,
                                             int output_nantennas,
                                             int nchans,
                                             int ntimestamps)
{
    // TAFTP to FTPA
    // Input dimensions
    int tp   = _config.nsamples_per_heap() * _config.npol();
    int ftp  = nchans * tp;
    int aftp = input_nantennas * ftp;

    // Output dimensions
    int pa  = _config.npol() * output_nantennas;
    int tpa = _config.nsamples_per_heap() * ntimestamps * pa;
    output.resize(nchans * tpa, {0, 0});

    for(int timestamp_idx = 0; timestamp_idx < ntimestamps; ++timestamp_idx) {
        for(int antenna_idx = 0; antenna_idx < input_nantennas; ++antenna_idx) {
            int input_antenna_idx = antenna_idx;
            for(int chan_idx = 0; chan_idx < nchans; ++chan_idx) {
                for(int samp_idx = 0; samp_idx < _config.nsamples_per_heap();
                    ++samp_idx) {
                    for(int pol_idx = 0; pol_idx < _config.npol(); ++pol_idx) {
                        int input_idx =
                            (timestamp_idx * aftp + input_antenna_idx * ftp +
                             chan_idx * tp + samp_idx * _config.npol() +
                             pol_idx);
                        int output_sample_idx =
                            timestamp_idx * _config.nsamples_per_heap() +
                            samp_idx;
                        int output_idx =
                            (chan_idx * tpa + output_sample_idx * pa +
                             pol_idx * output_nantennas + antenna_idx);
                        output[output_idx] = input[input_idx];
                    }
                }
            }
        }
    }
}

void TransposerTester::compare_against_host(DeviceVoltageType const& gpu_input,
                                            DeviceVoltageType const& gpu_output,
                                            std::size_t input_nantennas,
                                            std::size_t ntimestamps)
{
    HostVoltageType host_input = gpu_input;
    HostVoltageType host_output;
    HostVoltageType cuda_output = gpu_output;
    transpose_c_reference(host_input,
                          host_output,
                          input_nantennas,
                          _config.nantennas(),
                          _config.nchans(),
                          ntimestamps);
    for(int ii = 0; ii < host_output.size(); ++ii) {
        ASSERT_EQ(host_output[ii].x, cuda_output[ii].x);
        ASSERT_EQ(host_output[ii].y, cuda_output[ii].y);
    }
}

TEST_P(TransposerTester, cycling_prime_test)
{
    Transposer transposer(_config);
    std::size_t ntimestamps     = 12;
    std::size_t input_nantennas = GetParam();
    std::size_t input_size = (ntimestamps * input_nantennas * _config.nchans() *
                              _config.nsamples_per_heap() * _config.npol());

    HostVoltageType host_gpu_input(input_size);
    for(int ii = 0; ii < input_size; ++ii) {
        host_gpu_input[ii].x = (ii % 113);
        host_gpu_input[ii].y = (ii % 107);
    }
    DeviceVoltageType gpu_input = host_gpu_input;
    DeviceVoltageType gpu_output;
    transposer.transpose(gpu_input, gpu_output, input_nantennas, _stream);
    compare_against_host(gpu_input, gpu_output, input_nantennas, ntimestamps);
}

INSTANTIATE_TEST_SUITE_P(TransposerTesterSuite,
                         TransposerTester,
                         ::testing::Range<std::size_t>(1, SKYWEAVER_NANTENNAS, 1));

} // namespace test
} // namespace skyweaver
