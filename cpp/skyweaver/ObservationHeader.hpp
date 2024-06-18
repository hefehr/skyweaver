#ifndef SKYWEAVER_OBSERVATIONHEADER_HPP
#define SKYWEAVER_OBSERVATIONHEADER_HPP

#include "psrdada_cpp/raw_bytes.hpp"
#include "skyweaver/Header.hpp"

namespace skyweaver
{
struct ObservationHeader {
    std::size_t nchans             = 0;   // Number of frequency channels
    std::size_t npol               = 0;   // Number of polarisations
    std::size_t nbits              = 0;   // Number of bits per sample
    std::size_t nantennas          = 0;   // Number of antennas
    std::size_t sample_clock_start = 0;   // The start epoch in sampler ticks
    std::size_t chan0_idx          = 0;   // The index of the first channel
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
    std::string to_string() const;
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