#include "skyweaver/test/CoherentDedisperserTester.cuh"
#include "skyweaver/test/test_utils.cuh"
namespace skyweaver
{
namespace test
{

CoherentDedisperserTester::CoherentDedisperserTester(): ::testing::Test()
{
    pipeline_config.dada_header_size(4096);
    pipeline_config.dedisp_max_delay_samps(8192);
    pipeline_config.gulp_length_samps(32768);

}

CoherentDedisperserTester::~CoherentDedisperserTester()
{
}

void CoherentDedisperserTester::SetUp()
{
    dms.push_back(0);
    dms.push_back(100);
    dms.push_back(200);
    nantennas =  pipeline_config.nantennas();
    nchans = pipeline_config.nchans();
    npols = pipeline_config.npol();
    float f_low = 856.0;
    float bridge_bw = 13.375;
    float tsamp = 4.785046729e-6;

    float f1 = f_low;
    float f2 = f_low + 856.0/4096;
    float max_dm = *(std::max_element(dms.begin(), dms.end()));
    
    float max_dm_delay = CoherentDedisperser::get_dm_delay(f1,f2, max_dm);
    max_delay_samps = std::ceil(max_dm_delay / tsamp);
    
    BOOST_LOG_TRIVIAL(info) << "Max dm delay per subband is: " << max_dm_delay << " seconds which is about " << max_delay_samps << " samples." ;

    CoherentDedisperser::createConfig(dedisp_config,  pipeline_config.gulp_length_samps(), max_delay_samps, nchans, npols, nantennas,tsamp, f_low, bridge_bw, dms);

    
}

void CoherentDedisperserTester::TearDown()
{
}
TEST_F(CoherentDedisperserTester, testCoherentDedisperser)
{
    CoherentDedisperser coherentDedisperser(pipeline_config, dedisp_config);

    int nsamples_gulp = pipeline_config.gulp_length_samps();

    std::size_t max_delay_tpa = this->max_delay_samps * nantennas * npols;
    std::size_t block_length_tpa = nsamples_gulp * nantennas * npols;

    BOOST_LOG_TRIVIAL(info) << "max_delay_tpa: " << max_delay_tpa;
    BOOST_LOG_TRIVIAL(info) << "block_length_tpa: " << block_length_tpa;

    thrust::host_vector<char2> h_voltages;
    random_normal_complex(h_voltages,
       block_length_tpa,
        0.0f,
        17.0f);

    
    BOOST_LOG_TRIVIAL(info) << "h_voltages.size(): " << h_voltages.size();
    thrust::device_vector<char2> d_voltages(h_voltages.size());
    d_voltages = h_voltages;

    thrust::device_vector<char2> d_voltages_out(block_length_tpa - max_delay_tpa);
    BOOST_LOG_TRIVIAL(info) << "d_voltages_out.size(): " << d_voltages_out.size();


    std::size_t out_offset = 0;
    int dm_idx = 0;

    coherentDedisperser.dedisperse(d_voltages, d_voltages_out, out_offset, dm_idx);

    thrust::host_vector<char2> h_voltages_out = d_voltages_out;

    int acceptable_error =1;
    for(int i =0; i < h_voltages_out.size(); i++)
    {
        ASSERT_NEAR(h_voltages_out[i].x, h_voltages[max_delay_tpa/2+i].x, acceptable_error);
        ASSERT_NEAR(h_voltages_out[i].y, h_voltages[max_delay_tpa/2+i].y, acceptable_error);
    }


}

// TEST_F(CoherentDedisperserTester, testCoherentDedisperserBatchFFTs)
// {

//     CoherentDedisperser coherentDedisperser(pipeline_config, dedisp_config);
//     int nsamples_gulp = pipeline_config.gulp_length_samps();

//     thrust::host_vector<char2> h_t_voltages;
//     random_normal_complex(h_t_voltages,
//         nantennas * npols * (nsamples_gulp + pipeline_config.dedisp_max_delay_samps()),
//         0.0f,
//         17.0f);
    
//     // generate TPA voltages from T voltages above, by repeating each T for nantennas * npols times

//     thrust::host_vector<char2> h_tpa_voltages;
//     h_tpa_voltages.resize(nantennas * npols * nsamples_gulp);

//     for(int time =0; time < nsamples_gulp; time++)
//     {
//         for(int x =0; x < nantennas * npols; x++)
//         {
//             h_tpa_voltages[time * nantennas * npols + x] = h_t_voltages[time];
//         }
        
//     }

    
    

//     thrust::device_vector<char2> d_voltages(h_tpa_voltages.size());
//     d_voltages = h_tpa_voltages;

//     thrust::device_vector<char2> d_voltages_out(nantennas * npols * nsamples_gulp);
//     std::size_t out_offset = 0;
//     int dm_idx = 0;
//     coherentDedisperser.dedisperse(d_voltages, d_voltages_out, out_offset, dm_idx);
// }

} // namespace test
} // namespace skyweaver