#include "skyweaver/test/MultiFileReaderTester.cuh"

namespace skyweaver
{
namespace test
{

MultiFileReaderTester::MultiFileReaderTester(): ::testing::Test()
{
    std::vector<std::string> files;
    for(int i = 0; i < 10; i++) {
        std::string file_name =
            "data/dada_files/data_chunk_" + std::to_string(i) + ".dada";
        files.push_back(file_name);
    }
    pipeline_config.input_files(files);
    pipeline_config.dada_header_size(4096);
    multi_file_reader = std::make_unique<MultiFileReader>(pipeline_config);

    auto header = multi_file_reader->get_header();
    nantennas   = header.nantennas;
    nchans      = header.nchans;
    npols       = header.npol;
    nbits       = header.nbits;

    BOOST_LOG_TRIVIAL(debug) << "Header read,total filesize is:" << multi_file_reader->get_total_size();
    
}

MultiFileReaderTester::~MultiFileReaderTester()
{
}

void MultiFileReaderTester::SetUp()
{
}

void MultiFileReaderTester::TearDown()
{
}
TEST_F(MultiFileReaderTester, testMultiFileReader)
{
    std::string full_file_name = "data/dada_files/data.dada";

    std::size_t nsamples_multi_file =
        multi_file_reader->get_total_size() /
        (nantennas * nchans * npols * nbits * 2 / 8);
    BOOST_LOG_TRIVIAL(debug)
        << "nsamples from multi_file_reader: " << nsamples_multi_file;

    std::ifstream full_file(full_file_name, std::ios::binary);
    full_file.seekg(0, std::ios::end);
    int file_size = full_file.tellg();
    std::size_t nsamples_full_file =
        (file_size - pipeline_config.dada_header_size()) /
        (nantennas * nchans * npols * nbits * 2 / 8);
    BOOST_LOG_TRIVIAL(debug)
        << "nsamples from full_file_reader: " << nsamples_full_file;

    ASSERT_EQ(nsamples_full_file, nsamples_multi_file);

    int nsamples_gulp     = 4970;
    int char2s_per_sample = nantennas * nchans * npols;

    thrust::host_vector<char2> multi_file_voltages(char2s_per_sample *
                                                   nsamples_gulp);
    thrust::host_vector<char2> full_file_voltages(char2s_per_sample *
                                                  nsamples_gulp);

    BOOST_LOG_TRIVIAL(debug) << "Host vectors created";

    multi_file_reader->seekg(char2s_per_sample * nsamples_gulp, std::ios::beg);
    // BOOST_LOG_TRIVIAL(debug) << "Seek successful";

    typedef decltype(multi_file_voltages)::value_type valType;

    multi_file_reader->read(
        reinterpret_cast<char*>(
            thrust::raw_pointer_cast(multi_file_voltages.data())),
        static_cast<std::streamsize>(nantennas * nchans * npols *
                                     nsamples_gulp * sizeof(valType)));
    BOOST_LOG_TRIVIAL(debug) << "Multifile reader read data";

    full_file.seekg(pipeline_config.dada_header_size(), std::ios::beg);
    full_file.seekg(char2s_per_sample * nsamples_gulp, std::ios::cur);
    full_file.read(reinterpret_cast<char*>(
                       thrust::raw_pointer_cast(full_file_voltages.data())),
                   static_cast<std::streamsize>(nantennas * nchans * npols *
                                                nsamples_gulp* sizeof(valType)));

    for(int i = 0; i < multi_file_voltages.size(); i++) {
        ASSERT_EQ(multi_file_voltages[i].x, full_file_voltages[i].x);
        ASSERT_EQ(multi_file_voltages[i].y, full_file_voltages[i].y);
    }
}

TEST_F(MultiFileReaderTester, testMultiFileReaderEOF)
{
    // this should not give a run time error
    multi_file_reader->seekg(multi_file_reader->get_total_size(),
                             std::ios::beg);
    ASSERT_EQ(multi_file_reader->eof(), true);

    // this should give a run time error - test it
    ASSERT_THROW(
        multi_file_reader->seekg(multi_file_reader->get_total_size() + 1,
                                 std::ios::beg),
        std::runtime_error);
}

TEST_F(MultiFileReaderTester, testMultiFileReaderZeroSizeRead)
{
    thrust::host_vector<char2> multi_file_voltages(0);
    typedef decltype(multi_file_voltages)::value_type valType;

    ASSERT_EQ(multi_file_reader->read(
                  reinterpret_cast<char*>(
                      thrust::raw_pointer_cast(multi_file_voltages.data())),
                  0),
              0);
}

TEST_F(MultiFileReaderTester, testMultiFileReaderNonExistentFile)
{
    std::vector<std::string> files;
    files.push_back("data/dada_files/non_existent_file.dada");
    pipeline_config.input_files(files);
    ASSERT_THROW(std::make_unique<MultiFileReader>(pipeline_config),
                 std::runtime_error);
}

TEST_F(MultiFileReaderTester, testMultiFileReaderContiguity)
{
    std::vector<std::string> files;
    for(int i = 0; i < 10; i++) {
        std::string file_name =
            "data/dada_files/data_chunk_" + std::to_string(i) + ".dada";
        files.push_back(file_name);
    }
    pipeline_config.input_files(files);
    pipeline_config.dada_header_size(4096);
    pipeline_config.check_input_contiguity(true);
    ASSERT_NO_THROW(std::make_unique<MultiFileReader>(pipeline_config));
}

/**
check for contiguity
*/
} // namespace test
} // namespace skyweaver