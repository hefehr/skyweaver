#include "skyweaver/Header.hpp"
#include "skyweaver/MultiFileWriter.cuh"

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
STOKES       I

NBIT         8
NDIM         1
NPOL         1
NCHAN        64
NBEAM       800
ORDER        TFB

CHAN0_IDX 2688
)";

std::string get_formatted_time(long double unix_timestamp)
{
    char formatted_time[80];
    std::time_t unix_time = unix_timestamp;
    struct std::tm* ptm   = std::gmtime(&unix_time);
    strftime(formatted_time, sizeof(formatted_time), "%Y-%m-%d-%H:%M:%S", ptm);
    return std::string(formatted_time);
}

} // namespace

template <typename VectorType>
MultiFileWriter<VectorType>::MultiFileWriter(PipelineConfig const& config,
                                             std::string tag)
    : _config(config), _tag(tag)
{
}

template <typename VectorType>
MultiFileWriter<VectorType>::~MultiFileWriter(){};

template <typename VectorType>
void MultiFileWriter<VectorType>::init(ObservationHeader const& header)
{
    _header = header; // Make copy of the header
}

template <typename VectorType>
bool MultiFileWriter<VectorType>::has_stream(std::size_t stream_idx)
{
    auto it = _file_streams.find(stream_idx);
    return (it != _file_streams.end());
}

template <typename VectorType>
FileStream&
MultiFileWriter<VectorType>::create_stream(VectorType const& stream_data,
                                           std::size_t stream_idx)
{
    BOOST_LOG_TRIVIAL(debug) << "Creating stream based on stream prototype: "
                            << stream_data.describe();
    // Here we round the file size to a multiple of the stream prototype
    std::size_t filesize =
        std::max(1ul,
                 _config.max_output_filesize() / stream_data.size() /
                     sizeof(typename VectorType::value_type)) *
        stream_data.size();
    BOOST_LOG_TRIVIAL(debug)
        << "Maximum allowed file size = " << filesize << " bytes (+header)";

    _file_streams[stream_idx] = std::make_unique<FileStream>(
        get_output_dir(stream_data, stream_idx),
        get_basefilename(stream_data, stream_idx),
        get_extension(stream_data),
        filesize,
        [&, this, stream_data, stream_idx, filesize](
            std::size_t& header_size,
            std::size_t bytes_written,
            std::size_t file_idx) -> std::shared_ptr<char const> {
            header_size       = _config.dada_header_size();
            char* temp_header = new char[header_size];
            std::fill(temp_header, temp_header + header_size, 0);
            std::memcpy(temp_header,
                        default_dada_header.c_str(),
                        default_dada_header.size());
            psrdada_cpp::RawBytes bytes(temp_header,
                                        header_size,
                                        header_size,
                                        false);
            Header header_writer(bytes);
            header_writer.set<std::size_t>("NBEAM", stream_data.nbeams());
            header_writer.set<std::size_t>("NCHAN", stream_data.nchannels());
            header_writer.set<std::size_t>("OBS_NCHAN", _header.obs_nchans);
            header_writer.set<long double>("OBS_FREQUENCY",
                                           _header.obs_frequency);
            header_writer.set<long double>("OBS_BW", _header.obs_bandwidth);
            header_writer.set<std::size_t>("NSAMP", stream_data.nsamples());
            if(stream_data.ndms()) {
                header_writer.set<std::size_t>("NDMS", stream_data.ndms());
                header_writer.set("DMS", stream_data.dms(), 7);
            }
            header_writer.set<long double>(
                "COHERENT_DM",
                static_cast<long double>(stream_data.reference_dm()));
            header_writer.set<long double>("FREQ", _header.frequency);
            header_writer.set<long double>("BW", _header.bandwidth);
            header_writer.set<long double>("TSAMP", stream_data.tsamp() * 1e6);
            if(_config.stokes_mode() == "IQUV") {
                header_writer.set<std::size_t>("NPOL", 4);
            } else {
                header_writer.set<std::size_t>("NPOL", 1);
            }
            header_writer.set<std::string>("STOKES_MODE",
                                           _config.stokes_mode());
            header_writer.set<std::string>("ORDER",
                                           stream_data.dims_as_string());
            header_writer.set<std::size_t>("CHAN0_IDX", _header.chan0_idx);
            header_writer.set<std::size_t>("FILE_SIZE", filesize);
            header_writer.set<std::size_t>("FILE_NUMBER", file_idx);
            header_writer.set<std::size_t>("OBS_OFFSET", bytes_written);
            header_writer.set<std::size_t>("OBS_OVERLAP", 0);
            header_writer.set<long double>("UTC_START",
                                           _header.utc_start +
                                               stream_data.utc_offset());
            std::shared_ptr<char const> header_ptr(
                temp_header,
                std::default_delete<char[]>());
            return header_ptr;
        });
    return *_file_streams[stream_idx];
}

template <typename VectorType>
std::string 
MultiFileWriter<VectorType>::get_output_dir(VectorType const& stream_data,
                                              std::size_t stream_idx)
{
    // Output directory format
    // <utcstart>/<freq:%f.02>/<stream_id>
    std::stringstream output_dir;
    output_dir << _config.output_dir() << "/" 
               << get_formatted_time(_header.utc_start) << "/"
               << std::setprecision(9) << _header.frequency << "/" 
               << stream_idx << "/";
    return output_dir.str();
}

template <typename VectorType>
std::string
MultiFileWriter<VectorType>::get_basefilename(VectorType const& stream_data,
                                              std::size_t stream_idx)
{
    // Output file format
    // <prefix>_<utcstart>_<dm:%f.03>_<byte_offset>.<extension>
    std::stringstream base_filename;
    if(!_config.output_file_prefix().empty()) {
        base_filename << _config.output_file_prefix() << "_";
    }
    base_filename << get_formatted_time(_header.utc_start) << "_"
                  << stream_idx << "_" << std::fixed << std::setprecision(3)
                  << std::setfill('0') << std::setw(9)
                  << stream_data.reference_dm() << "_"
                  << std::setfill('0') << std::setw(9)
                  << _header.frequency;
    if(!_tag.empty()) {
        base_filename << "_" << _tag;
    }
    return base_filename.str();
}

template <typename VectorType>
std::string
MultiFileWriter<VectorType>::get_extension(VectorType const& stream_data)
{
    std::string dims = stream_data.dims_as_string();
    for(auto& c: dims) { c = std::tolower(static_cast<unsigned char>(c)); }
    return "." + dims;
}

template <typename VectorType>
bool MultiFileWriter<VectorType>::operator()(VectorType const& stream_data,
                                             std::size_t stream_idx)
{
    if(!has_stream(stream_idx)) {
        create_stream(stream_data, stream_idx);
    }
    if constexpr(is_device_vector<VectorType>::value) {
        thrust::host_vector<typename VectorType::value_type> stream_data_h =
            stream_data.vector();
        _file_streams.at(stream_idx)
            ->write(reinterpret_cast<char const*>(stream_data_h.data()),
                    stream_data_h.size() *
                        sizeof(typename VectorType::value_type));
    } else {
        _file_streams.at(stream_idx)
            ->write(reinterpret_cast<char const*>(thrust::raw_pointer_cast(stream_data.data())),
                    stream_data.size() *
                        sizeof(typename VectorType::value_type));
    }
    return false;
}

} // namespace skyweaver