
#include "skyweaver/ObservationHeader.hpp"

namespace skyweaver
{
void read_dada_header(psrdada_cpp::RawBytes& raw_header,
                      ObservationHeader& header)
{
    Header parser(raw_header);
    header.nchans     = parser.get<decltype(header.nchans)>("NCHAN");
    header.obs_nchans = parser.get<decltype(header.obs_nchans)>("OBS_NCHAN");
    header.npol       = parser.get<decltype(header.npol)>("NPOL");
    header.nbits      = parser.get<decltype(header.nbits)>("NBIT");
    header.nantennas  = parser.get<decltype(header.nantennas)>("NANT");
    header.sample_clock_start =
        parser.get<decltype(header.sample_clock_start)>("SAMPLE_CLOCK_START");
    header.bandwidth     = parser.get<decltype(header.bandwidth)>("BW");
    header.obs_bandwidth = parser.get<decltype(header.obs_bandwidth)>("OBS_BW");
    header.frequency     = parser.get<decltype(header.frequency)>("FREQ");
    header.obs_frequency =
        parser.get<decltype(header.obs_frequency)>("OBS_FREQ");
    header.tsamp = parser.get<decltype(header.tsamp)>("TSAMP");
    header.sample_clock =
        parser.get<decltype(header.sample_clock)>("SAMPLE_CLOCK");
    header.sync_time   = parser.get<decltype(header.sync_time)>("SYNC_TIME");
    header.utc_start   = parser.get<decltype(header.utc_start)>("UTC_START");
    header.mjd_start   = parser.get<decltype(header.mjd_start)>("MJD_START");
    header.source_name = parser.get<decltype(header.source_name)>("SOURCE");
    header.ra          = parser.get<decltype(header.ra)>("RA");
    header.dec         = parser.get<decltype(header.dec)>("DEC");
    header.telescope   = parser.get<decltype(header.telescope)>("TELESCOPE");
    header.instrument  = parser.get<decltype(header.instrument)>("INSTRUMENT");
    header.chan0_idx   = parser.get<decltype(header.chan0_idx)>("CHAN0_IDX");
    header.obs_offset  = parser.get<decltype(header.obs_offset)>("OBS_OFFSET");
}

void validate_header(ObservationHeader const& header,
                     PipelineConfig const& config)
{
    if(header.nantennas > config.nantennas()) {
        throw std::runtime_error(
            "Input data contains too many antennas for current build, "
            "-DSKYWEAVER_NANTENNAS should be larger than the data nantennas");
    }
    if(header.nchans != config.nchans()) {
        throw std::runtime_error(
            "Input data contains incorrect number of channels for current "
            "build, "
            "-DSKYWEAVER_NCHANS should be equal to the data nchans");
    }
    if(header.npol != config.npol()) {
        throw std::runtime_error(
            "Input data contains incorrect number of polarisations for current "
            "build, "
            "-DSKYWEAVER_NPOL should be equal to the data npol");
    }
    if(header.nbits != 8) {
        throw std::runtime_error(
            "Input data contains incorrect number of bits per sample, "
            "currently only 8-bit input supported");
    }
}

void update_config(PipelineConfig& config, ObservationHeader const& header)
{
    config.bandwidth(header.bandwidth);
    config.centre_frequency(header.frequency);
    // TO DO: might need to add other variables in the future.
}

std::string ObservationHeader::to_string() const
{
    std::ostringstream oss;
    oss << "ObservationHeader:\n"
        << "  nchans: " << nchans << "\n"
        << "  npol: " << npol << "\n"
        << "  nbits: " << nbits << "\n"
        << "  nantennas: " << nantennas << "\n"
        << "  sample_clock_start: " << sample_clock_start << "\n"
        << "  bandwidth (Hz): " << bandwidth << "\n"
        << "  centre frequency (Hz): " << frequency << "\n"
        << "  tsamp (s): " << tsamp << "\n"
        << "  sample_clock: " << sample_clock << "\n"
        << "  sync_time: " << sync_time << "\n"
        << "  utc_start: " << utc_start << "\n"
        << "  mjd_start: " << mjd_start << "\n"
        << "  source_name: " << source_name << "\n"
        << "  ra: " << ra << "\n"
        << "  dec: " << dec << "\n"
        << "  telescope: " << telescope << "\n"
        << "  instrument: " << instrument << "\n"
        << "  chan0_idx: " << chan0_idx << "\n"
        << "  obs_offset: " << obs_offset << "\n";

    return oss.str();
}

} // namespace skyweaver