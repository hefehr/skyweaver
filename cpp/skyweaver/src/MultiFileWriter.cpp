#include "skyweaver/MultiFileWriter.hpp"

#include "skyweaver/Header.hpp"

#include <cstring>
#include <ctime>
#include <iomanip>
#include <sstream>
#include <string>

/**
 * Now write a DADA file per DM
 * with optional time splitting
 */

namespace skyweaver
{

namespace
{

/**
 * The expectation is that each file contains data in
 * TBTF order. Need to explicitly update:
 * INNER_T --> the number of timesamples per block that was processed
 * NBEAMS --> the total number of beams in the file
 * DM --> the dispersion measure of this data
 * Freq --> the centre frequency of this subband (???)
 * BW --> the bandwidth of the subband (???)
 * TSAMP --> the sampling interval of the data
 * NPOL --> normally 1 but could be 4 for full stokes
 * CHAN0_IDX --> this comes from the obs header and uniquely identifies the
 * bridge
 */
std::string default_dada_header = R"(
HEADER       DADA
HDR_VERSION  1.0
HDR_SIZE     4096
DADA_VERSION 1.0

FILE_SIZE    100000000000
FILE_NUMBER  0

UTC_START    1708082229.000020336 
MJD_START    60356.47024305579093

SOURCE       J1644-4559
RA           16:44:49.27
DEC          -45:59:09.7
TELESCOPE    MeerKAT
INSTRUMENT   CBF-Feng
RECEIVER     L-band
FREQ         1284000000.000000
BW           856000000.000000
TSAMP        4.7850467290
DM           0.0
STOKES       I

NBIT         8
NDIM         1
NPOL         1
NCHAN        64
NBEAMS       800
ORDER        TBTF
INNER_T      256

CHAN0_IDX 2688
)";

} // namespace

MultiFileWriter::MultiFileWriter(PipelineConfig const& config, std::string tag)
: _config(config), _tag(tag)
{
    _file_streams.resize(_config.coherent_dms().size());
}
MultiFileWriter::~MultiFileWriter() {};

void MultiFileWriter::init(ObservationHeader const& header)
{
    std::size_t output_nchans = header.nchans / _config.cb_fscrunch();
    long double output_tsamp  = header.tsamp * _config.cb_tscrunch();
    BOOST_LOG_TRIVIAL(debug) << "Output Nchans = " << output_nchans;
    BOOST_LOG_TRIVIAL(debug) << "Output tsamp = " << output_tsamp << " s";

    // TODO this needs to include a 4 if running in full Stokes
    std::size_t btf_size =
        _config.nbeams() * _config.nsamples_per_block() * output_nchans;
    BOOST_LOG_TRIVIAL(debug) << "BTF size: " << btf_size;
    BOOST_LOG_TRIVIAL(debug)
        << "Requested max file size: " << _config.max_output_filesize();
    std::size_t filesize =
        (_config.max_output_filesize() / btf_size) * btf_size;
    if(filesize == 0)
        filesize = btf_size;
    BOOST_LOG_TRIVIAL(debug)
        << "Revised max size per file: " << filesize << " bytes";
    char formatted_time[80];
    std::time_t unix_time = header.utc_start;
    struct std::tm* ptm   = std::gmtime(&unix_time);
    strftime(formatted_time, 80, "%Y-%m-%d-%H:%M:%S", ptm);
    BOOST_LOG_TRIVIAL(debug) << "Timestamp for filename: " << formatted_time;

    for(std::size_t dm_idx = 0; dm_idx < _config.coherent_dms().size();
        ++dm_idx) {
        // Output file format
        // <prefix>_<utcstart>_<dm:%f.03>_<byte_offset>.tbtf/tbt
        std::stringstream base_filename;

        if(!_config.output_file_prefix().empty()) {
            base_filename << _config.output_file_prefix() << "_";
            if (!_tag.empty())
            {
                base_filename << _tag << "_";
            }
        }
        base_filename << formatted_time << "_" << std::fixed
                      << std::setprecision(3) << std::setfill('0')
                      << std::setw(9) << _config.coherent_dms()[dm_idx];
        _file_streams[dm_idx].reset(new FileStream(
            _config.output_dir(),
            base_filename.str(),
            ".tbtf",  // TODO: Update after incoherent dedispersion implemented
            filesize, // This has to be a BTF multiple
            [&, dm_idx, output_nchans, output_tsamp, filesize](
                std::size_t& header_size,
                std::size_t bytes_written,
                std::size_t file_idx) -> std::shared_ptr<char const> {
                // We do not explicitly delete[] this array
                // Cleanup is handled by the shared pointer
                // created below
                header_size       = 4096;
                char* temp_header = new char[header_size];
                psrdada_cpp::RawBytes bytes(temp_header,
                                            header_size,
                                            header_size,
                                            false);
                std::memcpy(temp_header,
                            default_dada_header.c_str(),
                            default_dada_header.size());
                Header header_writer(bytes);
                header_writer.set<std::size_t>("INNER_T",
                                               _config.nsamples_per_block());
                header_writer.set<std::size_t>("NBEAMS", _config.nbeams());
                BOOST_LOG_TRIVIAL(debug) << "dereferencing IDX " << dm_idx;
                BOOST_LOG_TRIVIAL(debug)
                    << "array size " << _config.coherent_dms().size();
                header_writer.set<long double>(
                    "DM",
                    static_cast<long double>(_config.coherent_dms()[dm_idx]));
                header_writer.set<long double>("FREQ", header.frequency);
                header_writer.set<long double>("BW", header.bandwidth);
                header_writer.set<long double>("TSAMP", output_tsamp);
                // TODO change when using full Stokes
                header_writer.set<std::size_t>("NPOL", 1);
                header_writer.set<std::size_t>("CHAN0_IDX", header.chan0_idx);
                header_writer.set<std::size_t>("FILE_SIZE", filesize);
                header_writer.set<std::size_t>("FILE_NUMBER", file_idx);
                header_writer.set<std::size_t>("OBS_OFFSET", bytes_written);
                header_writer.set<std::size_t>("OBS_OVERLAP", 0);
                /** Not needed when keeping the UTC fixed
                // TODO change when using full Stokes
                std::size_t nsamples_written = bytes_written / _config.nbeams()
                / output_nchans; long double delta_t_seconds = nsamples_written
                * output_tsamp; long double utc_start = header.utc_start() +
                delta_t_seconds; long double mjd_start = header.mjd_start() +
                delta_t_seconds / 86400.0; header_writer.set<long
                double>("UTC_START", header.chan0_idx());
                */
                std::shared_ptr<char const> header_ptr(
                    temp_header,
                    std::default_delete<char[]>());
                return header_ptr;
            }));
    }
}

} // namespace skyweaver