#include "skyweaver/test/CoherentDedisperserTester.cuh"
#include "skyweaver/test/test_utils.cuh"
namespace skyweaver
{
namespace test
{

CoherentDedisperserTester::CoherentDedisperserTester(): ::testing::Test()
{
}

CoherentDedisperserTester::~CoherentDedisperserTester()
{
}

void CoherentDedisperserTester::SetUp()
{
    dms.push_back(0);
    dms.push_back(1000);
    fft_length = 1024;
    // dms.push_back(200);
    nantennas        = 64;
    nchans           = 64;
    npols            = 2;
    double f_low     = 856.0;
    double bridge_bw = 13.375;
    double tsamp     = 4096 / 856e6;

    double f1     = f_low;
    double f2     = f_low + 856.0 / 4096;
    double max_dm = *(std::max_element(dms.begin(), dms.end()));

    double max_dm_delay = CoherentDedisperser::get_dm_delay(f1, f2, max_dm);
    max_delay_samps     = std::ceil(max_dm_delay / tsamp);

    BOOST_LOG_TRIVIAL(info)
        << "Max dm delay per subband is: " << max_dm_delay
        << " seconds which is about " << max_delay_samps << " samples.";

    CoherentDedisperser::createConfig(dedisp_config,
                                      fft_length,
                                      max_delay_samps,
                                      nchans,
                                      npols,
                                      nantennas,
                                      tsamp,
                                      f_low,
                                      bridge_bw,
                                      dms);
}

void CoherentDedisperserTester::TearDown()
{
}
TEST_F(CoherentDedisperserTester, testCoherentDedisperser)
{
    CoherentDedisperser coherentDedisperser(dedisp_config);

    std::size_t max_delay_tpa    = this->max_delay_samps * nantennas * npols;
    std::size_t block_length_tpa = fft_length * nantennas * npols;

    BOOST_LOG_TRIVIAL(info) << "max_delay_tpa: " << max_delay_tpa;
    BOOST_LOG_TRIVIAL(info) << "block_length_tpa: " << block_length_tpa;

    thrust::host_vector<char2> h_voltages;
    random_normal_complex(h_voltages, block_length_tpa, 0.0f, 17.0f);

    BOOST_LOG_TRIVIAL(info)
        << "testCoherentDedisperser input h_voltages.size(): "
        << h_voltages.size();
    thrust::device_vector<char2> d_voltages(h_voltages.size());
    d_voltages = h_voltages;

    thrust::device_vector<char2> d_voltages_out(block_length_tpa -
                                                max_delay_tpa);
    BOOST_LOG_TRIVIAL(info)
        << "testCoherentDedisperser output d_voltages_out.size(): "
        << d_voltages_out.size();

    unsigned int freq_idx = 0;
    unsigned int dm_idx   = 0;

    coherentDedisperser.dedisperse(d_voltages,
                                   d_voltages_out,
                                   freq_idx,
                                   dm_idx);

    thrust::host_vector<char2> h_voltages_out = d_voltages_out;

    int acceptable_error = 1;
    for(int i = 0; i < h_voltages_out.size(); i++) {
        ASSERT_NEAR(h_voltages_out[i].x,
                    h_voltages[max_delay_tpa / 2 + i].x,
                    acceptable_error);
        ASSERT_NEAR(h_voltages_out[i].y,
                    h_voltages[max_delay_tpa / 2 + i].y,
                    acceptable_error);
    }
}

TEST_F(CoherentDedisperserTester, testCoherentDedisperserWithPythonData)
{
    CoherentDedisperser coherentDedisperser(dedisp_config);

    std::size_t max_delay_tpa    = this->max_delay_samps * nantennas * npols;
    std::size_t block_length_tpa = fft_length * nantennas * npols;

    BOOST_LOG_TRIVIAL(info) << "max_delay_tpa: " << max_delay_tpa;
    BOOST_LOG_TRIVIAL(info) << "block_length_tpa: " << block_length_tpa;
    int dm_idx = 1;
    float dm   = dms[dm_idx];

    /*************** Get input voltages ******************/

    // input file name is of format codedisp_input_DM<dm>.dat build the filename
    // here
    std::string inp_filename =
        "/homes/vkrishnan/dev/beamformer/skyweaver/cpp/skyweaver/test/data/"
        "dedispersion/codedisp_input_DM" +
        std::to_string(int(dm)) + "_1chan.bin";
    std::string out_filename =
        "/homes/vkrishnan/dev/beamformer/skyweaver/cpp/skyweaver/test/data/"
        "dedispersion/codedisp_output_DM" +
        std::to_string(int(dm)) + "_1chan.bin";

    std::ifstream codedisp_input(inp_filename, std::ios::binary);
    EXPECT_TRUE(codedisp_input.is_open());

    codedisp_input.seekg(0, std::ios::end);
    std::size_t inp_filesize = codedisp_input.tellg();
    codedisp_input.seekg(0, std::ios::beg);
    BOOST_LOG_TRIVIAL(info) << "Input File size: " << inp_filesize;

    // read entire file to host vector
    thrust::host_vector<char2> h_voltages;
    h_voltages.resize(inp_filesize / sizeof(char2));
    codedisp_input.read(reinterpret_cast<char*>(h_voltages.data()),
                        inp_filesize);
    BOOST_LOG_TRIVIAL(debug)
        << "Input h_voltages.size(): " << h_voltages.size();

    /********************************************************************/
    /*************** Get output voltages ******************/

    std::ifstream codedisp_output(out_filename, std::ios::binary);
    EXPECT_TRUE(codedisp_output.is_open());

    codedisp_output.seekg(0, std::ios::end);
    std::size_t out_filesize = codedisp_output.tellg();
    codedisp_output.seekg(0, std::ios::beg);
    BOOST_LOG_TRIVIAL(info) << "Python output file size: " << out_filesize;

    // read entire file to host vector
    thrust::host_vector<char2> h_voltages_check;
    h_voltages_check.resize(out_filesize / sizeof(char2));
    codedisp_output.read(reinterpret_cast<char*>(h_voltages_check.data()),
                         out_filesize);
    BOOST_LOG_TRIVIAL(debug)
        << "Python output h_voltages_check.size(): " << h_voltages_check.size();

    /********************************************************************/
    /************** Run the dedisperser and compare results ************/

    thrust::device_vector<char2> d_voltages(h_voltages_check.size());
    d_voltages = h_voltages;

    thrust::device_vector<char2> d_voltages_out(block_length_tpa -
                                                max_delay_tpa);
    BOOST_LOG_TRIVIAL(info)
        << "cpp output d_voltages_out.size(): " << d_voltages_out.size();

    unsigned int freq_idx = 0;

    coherentDedisperser.dedisperse(d_voltages,
                                   d_voltages_out,
                                   freq_idx,
                                   dm_idx);
    thrust::host_vector<char2> h_voltages_out = d_voltages_out;

    int acceptable_error = 1;
    for(int i = 0; i < h_voltages_out.size(); i++) {
        ASSERT_NEAR(h_voltages_out[i].x,
                    h_voltages_check[i].x,
                    acceptable_error);
        ASSERT_NEAR(h_voltages_out[i].y,
                    h_voltages_check[i].y,
                    acceptable_error);
    }
}

TEST_F(CoherentDedisperserTester, testCoherentDedisperserWithPythonDataAllChans)
{
    CoherentDedisperser coherentDedisperser(dedisp_config);

    std::size_t max_delay_tpa    = this->max_delay_samps * nantennas * npols;
    std::size_t block_length_tpa = fft_length * nantennas * npols;

    BOOST_LOG_TRIVIAL(info) << "max_delay_tpa: " << max_delay_tpa;
    BOOST_LOG_TRIVIAL(info) << "block_length_tpa: " << block_length_tpa;
    int dm_idx = 1;
    float dm   = dms[dm_idx];

    /*************** Get input voltages ******************/

    // input file name is of format codedisp_input_DM<dm>.dat build the filename
    // here
    std::string inp_filename =
        "/homes/vkrishnan/dev/beamformer/skyweaver/cpp/skyweaver/test/data/"
        "dedispersion/codedisp_input_DM" +
        std::to_string(int(dm)) + ".bin";
    std::string out_filename =
        "/homes/vkrishnan/dev/beamformer/skyweaver/cpp/skyweaver/test/data/"
        "dedispersion/codedisp_output_DM" +
        std::to_string(int(dm)) + ".bin";

    std::ifstream codedisp_input(inp_filename, std::ios::binary);
    EXPECT_TRUE(codedisp_input.is_open());

    codedisp_input.seekg(0, std::ios::end);
    std::size_t inp_filesize = codedisp_input.tellg();
    codedisp_input.seekg(0, std::ios::beg);
    BOOST_LOG_TRIVIAL(info) << "codedisp_input File size: " << inp_filesize;

    // read entire file to host vector
    thrust::host_vector<char2> h_voltages;
    h_voltages.resize(inp_filesize / sizeof(char2));
    codedisp_input.read(reinterpret_cast<char*>(h_voltages.data()),
                        inp_filesize);
    BOOST_LOG_TRIVIAL(debug)
        << "codedisp_input h_voltages.size(): " << h_voltages.size();

    /********************************************************************/
    /*************** Get output voltages ******************/

    std::ifstream codedisp_output(out_filename, std::ios::binary);
    EXPECT_TRUE(codedisp_output.is_open());

    codedisp_output.seekg(0, std::ios::end);
    std::size_t out_filesize = codedisp_output.tellg();
    codedisp_output.seekg(0, std::ios::beg);
    BOOST_LOG_TRIVIAL(info) << "codedisp_output File size: " << out_filesize;

    // read entire file to host vector
    thrust::host_vector<char2> h_voltages_check;
    h_voltages_check.resize(out_filesize / sizeof(char2));
    codedisp_output.read(reinterpret_cast<char*>(h_voltages_check.data()),
                         out_filesize);
    BOOST_LOG_TRIVIAL(debug)
        << "Python output h_voltages_check.size(): " << h_voltages_check.size();

    /********************************************************************/
    /************** Run the dedisperser and compare results ************/

    thrust::device_vector<char2> d_voltages(h_voltages_check.size());
    d_voltages = h_voltages;

    thrust::device_vector<char2> d_voltages_out(h_voltages_check.size());
    BOOST_LOG_TRIVIAL(debug)
        << "cpp output d_voltages_out.size(): " << d_voltages_out.size();

    for(int i = 0; i < nchans; i++) {
        // becasuse the vector's length is used in calculations inside the code.
        // It cannot take FTPA

        thrust::device_vector<char2> d_tpa_voltages(block_length_tpa);
        thrust::copy(d_voltages.begin() + i * block_length_tpa,
                     d_voltages.begin() + (i + 1) * block_length_tpa,
                     d_tpa_voltages.begin());
        unsigned int freq_idx = i;
        unsigned int dm_idx   = 1;
        coherentDedisperser.dedisperse(d_tpa_voltages,
                                       d_voltages_out,
                                       freq_idx,
                                       dm_idx);
    }

    thrust::host_vector<char2> h_voltages_out = d_voltages_out;

    int acceptable_error = 1;
    for(int i = 0; i < h_voltages_out.size(); i++) {
        ASSERT_NEAR(h_voltages_out[i].x,
                    h_voltages_check[i].x,
                    acceptable_error);
        ASSERT_NEAR(h_voltages_out[i].y,
                    h_voltages_check[i].y,
                    acceptable_error);
    }
}

TEST_F(CoherentDedisperserTester, testCoherentDedisperserChirpMultiply)
{
    int fft_length       = 1024;
    int NCHANS_PER_BLOCK = 128;

    thrust::host_vector<cufftComplex> h_voltages;
    h_voltages.resize(fft_length);
    random_normal_complex(h_voltages, fft_length, 0.0f, 17.0f);

    thrust::device_vector<cufftComplex> d_voltages = h_voltages;

    thrust::host_vector<cufftComplex> h_chirp(fft_length,
                                              make_cuComplex(1.0, 1.0));
    random_normal_complex(h_chirp, fft_length, 0.0f, 17.0f);
    thrust::device_vector<cufftComplex> d_chirp = h_chirp;

    thrust::host_vector<cufftComplex> h_voltages_out_check(h_voltages.size());

    for(int i = 0; i < fft_length; i++) {
        float a                   = h_voltages[i].x;
        float b                   = h_voltages[i].y;
        float c                   = h_chirp[i].x;
        float d                   = h_chirp[i].y;
        h_voltages_out_check[i].x = a * c - b * d;
        h_voltages_out_check[i].y = a * d + b * c;
    }

    thrust::device_vector<cufftComplex> d_voltages_out(h_voltages.size());

    dim3 blockSize(1 * 1);
    dim3 gridSize(fft_length / NCHANS_PER_BLOCK);
    kernels::dedisperse<<<gridSize, blockSize>>>(
        thrust::raw_pointer_cast(d_chirp.data()),
        thrust::raw_pointer_cast(d_voltages.data()),
        thrust::raw_pointer_cast(d_voltages_out.data()),
        fft_length);

    thrust::host_vector<cufftComplex> h_voltages_out = d_voltages_out;

    for(int i = 0; i < h_voltages_out.size(); i++) {
        ASSERT_NEAR(h_voltages_out[i].x, h_voltages_out_check[i].x, 0.001);
        ASSERT_NEAR(h_voltages_out[i].y, h_voltages_out_check[i].y, 0.001);
    }
}

TEST_F(CoherentDedisperserTester, testCoherentDedisperserChirpMultiply64x2)
{
    int fft_length       = 1024;
    int NCHANS_PER_BLOCK = 128;
    int nantennas        = 64;
    int npols            = 2;

    std::size_t vector_size = fft_length * nantennas * npols;

    thrust::host_vector<cufftComplex> h_voltages;
    h_voltages.resize(vector_size);
    random_normal_complex(h_voltages, vector_size, 0.0f, 17.0f);

    thrust::device_vector<cufftComplex> d_voltages = h_voltages;

    thrust::host_vector<cufftComplex> h_chirp(fft_length,
                                              make_cuComplex(1.0, 1.0));
    random_normal_complex(h_chirp, fft_length, 0.0f, 17.0f);
    thrust::device_vector<cufftComplex> d_chirp = h_chirp;

    thrust::host_vector<cufftComplex> h_voltages_out_check(vector_size);

    for(int t = 0; t < fft_length; t++) {
        float c = h_chirp[t].x;
        float d = h_chirp[t].y;
        for(int pa = 0; pa < nantennas * npols; pa++) {
            int idx                     = t * nantennas * npols + pa;
            float a                     = h_voltages[idx].x;
            float b                     = h_voltages[idx].y;
            h_voltages_out_check[idx].x = a * c - b * d;
            h_voltages_out_check[idx].y = a * d + b * c;
        }
    }

    thrust::device_vector<cufftComplex> d_voltages_out(vector_size);

    dim3 blockSize(nantennas * npols);
    dim3 gridSize(fft_length / NCHANS_PER_BLOCK);

    kernels::dedisperse<<<gridSize, blockSize>>>(
        thrust::raw_pointer_cast(d_chirp.data()),
        thrust::raw_pointer_cast(d_voltages.data()),
        thrust::raw_pointer_cast(d_voltages_out.data()),
        fft_length); // 1024

    thrust::host_vector<cufftComplex> h_voltages_out = d_voltages_out;

    for(int i = 0; i < h_voltages_out.size(); i++) {
        ASSERT_NEAR(h_voltages_out[i].x, h_voltages_out_check[i].x, 0.001);
        ASSERT_NEAR(h_voltages_out[i].y, h_voltages_out_check[i].y, 0.001);
    }
}

} // namespace test
} // namespace skyweaver