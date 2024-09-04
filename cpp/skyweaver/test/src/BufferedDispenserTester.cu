#include "skyweaver/BufferedDispenser.cuh"
#include "skyweaver/PipelineConfig.hpp"
#include "skyweaver/test/BufferedDispenserTester.cuh"
#include "skyweaver/test/test_utils.cuh"
#include "skyweaver/dedispersion_utils.cuh"

#include <random>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

namespace skyweaver
{
namespace test
{
BufferedDispenserTester::BufferedDispenserTester(): ::testing::Test()
{
    pipeline_config.gulp_length_samps(64);
    std::vector<float> dms;
    dms.push_back(0);
    dms.push_back(1000);
    std::size_t fft_length   = 1024;
    double tsamp = 4096 / 856e6;

    // dms.push_back(200);
    nantennas        = 64;
    nchans           = 64;
    npols            = 2;
    double f_low     = 856.0 * 1e6;
    double bridge_bw = 13.375 * 1e6;

    double f1 = f_low;
    double f2 = f_low + 856.0 * 1e6 / 4096;

    double max_dm = *(std::max_element(dms.begin(), dms.end()));
    BOOST_LOG_TRIVIAL(info) << "Max DM: " << max_dm;
    BOOST_LOG_TRIVIAL(info) << "tsamp: " << tsamp;
    BOOST_LOG_TRIVIAL(info) << "f_low: " << f_low;
    BOOST_LOG_TRIVIAL(info) << "f1: " << f1;
    BOOST_LOG_TRIVIAL(info) << "f2: " << f2;
    double max_dm_delay = dm_delay(f1, f2, max_dm);
    max_delay_samps     = std::ceil(max_dm_delay / tsamp);

    BOOST_LOG_TRIVIAL(info)
        << "Max dm delay per subband is: " << max_dm_delay
        << " seconds which is about " << max_delay_samps << " samples.";

    create_coherent_dedisperser_config(dedisp_config,
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

BufferedDispenserTester::~BufferedDispenserTester()
{
}

void BufferedDispenserTester::SetUp()
{
}

void BufferedDispenserTester::TearDown()
{
}

TEST_F(BufferedDispenserTester, testBufferedDispenser)
{
    // std::vector<float> coherent_dms;
    // coherent_dms.push_back(0.0);
    BufferedDispenser bufferedDispenser(pipeline_config, dedisp_config, nullptr);

    nantennas         = pipeline_config.nantennas();
    nchans            = pipeline_config.nchans();
    npols             = pipeline_config.npol();
    int nsamples_gulp = pipeline_config.gulp_length_samps();

    /*generate char2 voltages_1 */

    typename BufferedDispenser::FTPAVoltagesH voltages_1h(
        {static_cast<std::size_t>(nchans),
         static_cast<std::size_t>(nsamples_gulp),
         static_cast<std::size_t>(npols),
         static_cast<std::size_t>(nantennas)});
    voltages_1h.frequencies(pipeline_config.channel_frequencies());
    random_normal_complex(voltages_1h.vector(),
                          nantennas * nchans * npols * nsamples_gulp,
                          0.0f,
                          17.0f);
    ASSERT_EQ(voltages_1h.size(), nantennas * nchans * npols * nsamples_gulp);
    typename BufferedDispenser::FTPAVoltagesTypeD voltages_1 = voltages_1h;

    const std::size_t size_TPA = nantennas * npols * nsamples_gulp;
    const std::size_t max_delay_tpa =
        max_delay_samps * nantennas * npols;
    const std::size_t size_tpa = size_TPA + max_delay_tpa;

    /* hoard the voltages_1 */
    bufferedDispenser.hoard(voltages_1); // FTPA voltages_1
    for(int i = 0; i < nchans; i++) {
        auto const& dispensed_voltages_1 = bufferedDispenser.dispense(
            i); // tPA voltages_1 = overlap + T*PA voltages_1
        ASSERT_EQ(dispensed_voltages_1.size(), size_tpa);

        typename BufferedDispenser::HostTPAVoltagesH dispensed_voltages_1h =
            dispensed_voltages_1;

        std::size_t start_idx = max_delay_tpa; // start from end of overlap
        int k                 = 0;
        for(std::size_t j = start_idx; j < dispensed_voltages_1.size(); j++) {
            ASSERT_EQ(voltages_1h[i * size_TPA + k].x,
                      dispensed_voltages_1h[j].x);
            ASSERT_EQ(voltages_1h[i * size_TPA + k].y,
                      dispensed_voltages_1h[j].y);
            k++;
        }
        for(std::size_t j = 0; j < start_idx; j++) {
            ASSERT_EQ(0, dispensed_voltages_1h[j].x);
            ASSERT_EQ(0, dispensed_voltages_1h[j].y);
        }
    }

    /*generate char2 voltages_2 */
    typename BufferedDispenser::FTPAVoltagesH voltages_2h(
        {static_cast<std::size_t>(nchans),
         static_cast<std::size_t>(nsamples_gulp),
         static_cast<std::size_t>(npols),
         static_cast<std::size_t>(nantennas)});
    voltages_2h.frequencies(pipeline_config.channel_frequencies());
    random_normal_complex(voltages_2h.vector(),
                          nantennas * nchans * npols * nsamples_gulp,
                          0.0f,
                          17.0f);
    typename BufferedDispenser::FTPAVoltagesTypeD voltages_2 = voltages_2h;
    bufferedDispenser.hoard(voltages_2); // FTPA voltages_1
    for(int i = 0; i < nchans; i++) {
        auto const& dispensed_voltages_2 = bufferedDispenser.dispense(
            i); // tPA voltages_1 = overlap + T*PA voltages_1
        ASSERT_EQ(dispensed_voltages_2.size(), size_tpa);
        typename BufferedDispenser::HostTPAVoltagesH dispensed_voltages_2h =
            dispensed_voltages_2;
        std::size_t start_idx = max_delay_tpa; // start from end of overlap
        int k                 = 0;
        for(std::size_t j = start_idx; j < dispensed_voltages_2.size(); j++) {
            ASSERT_EQ(voltages_2h[i * size_TPA + k].x,
                      dispensed_voltages_2h[j].x);
            ASSERT_EQ(voltages_2h[i * size_TPA + k].y,
                      dispensed_voltages_2h[j].y);
            k++;
        }
        for(std::size_t j = 0; j < start_idx; j++) {
            ASSERT_EQ(voltages_1h[(i + 1) * size_TPA - max_delay_tpa + j].x,
                      dispensed_voltages_2h[j].x);
            ASSERT_EQ(voltages_1h[(i + 1) * size_TPA - max_delay_tpa + j].y,
                      dispensed_voltages_2h[j].y);
        }
    }
}
} // namespace test
} // namespace skyweaver