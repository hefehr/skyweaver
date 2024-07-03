#include "skyweaver/test/MultiFileWriterTester.cuh"

#include <cstdio>
#include <vector>
#include <filesystem>
#include <iostream>
#include <stdlib.h>
#include <string>

namespace fs = std::filesystem;

namespace skyweaver
{
namespace test
{

MultiFileWriterTester::MultiFileWriterTester(): ::testing::Test()
{
}

MultiFileWriterTester::~MultiFileWriterTester()
{
}

void MultiFileWriterTester::SetUp()
{
    char template_dirname[] = "/tmp/skyweaver_test_XXXXXX";
    char* directory_path    = mkdtemp(template_dirname);
    _config.output_dir(std::string(directory_path));
    std::vector<float> dms;
    for(float dm = 0.0f; dm < 5; dm += 1.2345f) { dms.push_back(dm); }
    _config.coherent_dms(dms);
}

void MultiFileWriterTester::TearDown()
{
    fs::remove_all(_config.output_dir());
}

TEST_F(MultiFileWriterTester, simple_updating_write)
{
    BOOST_LOG_TRIVIAL(debug)
        << "Testing in tmp directory: " << _config.output_dir();
    MultiFileWriter mfw(_config);
    _config.max_output_filesize(1000);
    std::size_t output_nchans = _config.nchans() / _config.cb_fscrunch();
    // TODO this needs to include a 4 if running in full Stokes
    std::size_t btf_size =
        _config.nbeams() * _config.nsamples_per_block() * output_nchans;
    std::vector<int8_t> powers(btf_size);
    ObservationHeader header;
    header.nchans = _config.nchans();
    header.tsamp  = 0.000064;
    BOOST_LOG_TRIVIAL(debug) << "HEADER: \n" << header.to_string();
    mfw.init(header);
    for(std::size_t file_idx = 0; file_idx < 4; ++file_idx) {
        for(std::size_t dm_idx = 0; dm_idx < _config.coherent_dms().size();
            ++dm_idx) {
            mfw(powers, dm_idx);
        }
    }
}

} // namespace test
} // namespace skyweaver