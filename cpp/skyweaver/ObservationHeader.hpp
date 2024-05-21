#ifndef SKYWEAVER_OBSERVATIONHEADER_HPP
#define SKYWEAVER_OBSERVATIONHEADER_HPP

#include "skyweaver/Header.hpp"
#include "psrdada_cpp/raw_bytes.hpp"

namespace skyweaver
{

/** --- example header ---

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
 */

struct ObservationHeader {
    std::size_t nchans             = 0;   // Number of frequency channels
    std::size_t npol               = 0;   // Number of polarisations
    std::size_t nbits              = 0;   // Number of bits per sample
    std::size_t nantennas          = 0;   // Number of antennas
    std::size_t sample_clock_start = 0;   // The start epoch in sampler ticks
    long double bandwidth          = 0.0; // Bandwidth in Hz
    long double frequency          = 0.0; // Centre frequency in Hz
    long double tsamp              = 0.0; // Sampling interval in microseconds
    long double sample_clock       = 0.0; // The sampling rate in Hz
    long double sync_time          = 0.0; // The UNIX epoch of the sampler zero
    long double utc_start          = 0.0; // The UTC start time of the data
    long double mjd_start          = 0.0; // The MJD start time of the data
    std::string source_name;              // Name of observation target
    std::string ra;                       // Right ascension
    std::string dec;                      // Declination
    std::string telescope;                // Telescope name
    std::string instrument;               // Name of the recording instrument
};

/**
 * @brief Parse header information for a DADA header block
 * 
 * @param raw_header A RawBytes object containing the DADA header
 * @param header An ObservationHeader instance
 */
void read_dada_header(psrdada_cpp::RawBytes& raw_header,
                      ObservationHeader& header);

} // namespace skyweaver

#endif // SKYWEAVER_OBSERVATIONHEADER_HPP