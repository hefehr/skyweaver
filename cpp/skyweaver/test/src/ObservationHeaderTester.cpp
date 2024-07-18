#include "skyweaver/test/ObservationHeaderTester.hpp"

namespace skyweaver
{
namespace test
{

ObservationHeaderTester::ObservationHeaderTester(): ::testing::Test()
{
}

ObservationHeaderTester::~ObservationHeaderTester()
{
}

void ObservationHeaderTester::SetUp()
{
}

void ObservationHeaderTester::TearDown()
{
}

TEST_F(ObservationHeaderTester, test_header_read)
{
    char header_bytes[4096] = R""""(
HEADER       DADA
HDR_VERSION  1.0
HDR_SIZE     4096
DADA_VERSION 1.0

# DADA parameters
OBS_ID       test

FILE_SIZE    956305408
FILE_NUMBER  0

# time of the rising edge of the first time sample
UTC_START    1708082229.000020336
MJD_START    60356.47024305579093

OBS_OFFSET   0
OBS_OVERLAP  0

SOURCE       J1644-4559
RA           16:44:49.27
DEC          -45:59:09.7
TELESCOPE    MeerKAT
INSTRUMENT   CBF-Feng
RECEIVER     L-band
FREQ         1284000000.000000
BW           856000000.000000
TSAMP        4.7850467290

BYTES_PER_SECOND 3424000000.000000

NBIT         8
NDIM         2
NPOL         2
NCHAN        64
NANT         57
ORDER        TAFTP
INNER_T      256

#MeerKAT specifics
SYNC_TIME    1708039531.000000
SAMPLE_CLOCK 1712000000.000000
SAMPLE_CLOCK_START 73098976034816
CHAN0_IDX 2688
OBS_FREQ     1284000000.000000
OBS_BW       856000000.000000
OBS_NCHAN    4096
    )"""";
    psrdada_cpp::RawBytes raw_header(header_bytes, 4096, 4096, false);
    ObservationHeader header;
    read_dada_header(raw_header, header);
    ASSERT_EQ(header.nantennas, 57);
    ASSERT_EQ(header.nbits, 8);
    ASSERT_EQ(header.nchans, 64);
    ASSERT_EQ(header.npol, 2);
    ASSERT_EQ(header.sample_clock_start, 73098976034816);
    ASSERT_DOUBLE_EQ(header.sample_clock, 1712000000.000000);
    ASSERT_DOUBLE_EQ(header.sync_time, 1708039531.000000);
    ASSERT_DOUBLE_EQ(header.utc_start, 1708082229.000020336);
    ASSERT_DOUBLE_EQ(header.mjd_start, 60356.47024305579093);
    ASSERT_EQ(header.source_name, "J1644-4559");
    ASSERT_EQ(header.ra, "16:44:49.27");
    ASSERT_EQ(header.dec, "-45:59:09.7");
    ASSERT_EQ(header.telescope, "MeerKAT");
    ASSERT_EQ(header.instrument, "CBF-Feng");
}

} // namespace test
} // namespace skyweaver
