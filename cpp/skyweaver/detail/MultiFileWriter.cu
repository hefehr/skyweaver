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

std::string get_formatted_time(long double unix_timestamp)
{
    char formatted_time[80];
    std::time_t unix_time = unix_timestamp;
    struct std::tm* ptm   = std::gmtime(&unix_time);
    strftime(formatted_time, sizeof(formatted_time), "%Y-%m-%d-%H:%M:%S", ptm);
    return std::string(formatted_time);
}

} // namespace


template <typename T, typename Enable = void> struct HeaderFormatter;

template <typename T>
struct HeaderFormatter<T, typename std::enable_if_t <
    std::is_same<T, FPAStatsH<typename T::value_type>>::value || 
    std::is_same<T, FPAStatsD<typename T::value_type>>::value>>
{
    void operator()(T const& stream_data, 
                    ObservationHeader const& obs_header,
                    PipelineConfig const& config,
                    Header& output_header){
        output_header.set<std::size_t>("NCHAN", stream_data.nchannels());
        output_header.set<std::size_t>("NSAMP", stream_data.nsamples());
        output_header.set<std::string>("STOKES_MODE", "I");
        output_header.set<std::size_t>("NPOL", stream_data.npol());
        output_header.set<std::size_t>("NDIM", 1);
        output_header.set<std::size_t>("NBIT", 32);
        output_header.set<std::size_t>("NANT", stream_data.nantennas());
        output_header.set<std::string>("DTYPE", "MOMENTS");       
    }
};

template <typename T> 
struct HeaderFormatter<T, typename std::enable_if_t <
    std::is_same<T, TDBPowersH<typename T::value_type>>::value || 
    std::is_same<T, TDBPowersD<typename T::value_type>>::value>>
{
    void operator()(T const& stream_data, 
                    ObservationHeader const& obs_header,
                    PipelineConfig const& config,
                    Header& output_header){
        output_header.set<std::size_t>("NCHAN", stream_data.nchannels());
        output_header.set<std::size_t>("NSAMP", stream_data.nsamples());
        output_header.set<std::size_t>("NBEAM", stream_data.nbeams());
        output_header.set<std::size_t>("NDMS", stream_data.ndms());
        output_header.set("DMS", stream_data.dms(), 7);
        output_header.set<long double>("COHERENT_DM",
                static_cast<long double>(stream_data.reference_dm()));
        output_header.set<std::string>("STOKES_MODE", config.stokes_mode());
        output_header.set<std::size_t>("NPOL", config.stokes_mode().size());
        output_header.set<std::size_t>("NDIM", 1);
        output_header.set<std::size_t>("NBIT", sizeof(typename T::value_type) * 8);
        output_header.set<std::string>("DTYPE", "INTENSITIES");     
    }
};

template <typename T> 
struct HeaderFormatter<T, typename std::enable_if_t <
    std::is_same<T, TFBPowersH<typename T::value_type>>::value || 
    std::is_same<T, TFBPowersD<typename T::value_type>>::value>>
{
    void operator()(T const& stream_data, 
                    ObservationHeader const& obs_header,
                    PipelineConfig const& config,
                    Header& output_header){
        output_header.set<std::size_t>("NCHAN", stream_data.nchannels());
        output_header.set<std::size_t>("NSAMP", stream_data.nsamples());
        output_header.set<std::size_t>("NBEAM", stream_data.nbeams());
        output_header.set<long double>("COHERENT_DM",
                static_cast<long double>(stream_data.reference_dm()));
        output_header.set<std::string>("STOKES_MODE", config.stokes_mode());
        output_header.set<std::size_t>("NPOL", config.stokes_mode().size());
        output_header.set<std::size_t>("NDIM", 1);
        output_header.set<std::size_t>("NBIT", sizeof(typename T::value_type) * 8);
        output_header.set<std::string>("DTYPE", "INTENSITIES");  
    }
};

template <typename T> 
struct HeaderFormatter<T, typename std::enable_if_t <
    std::is_same<T, BTFPowersH<typename T::value_type>>::value || 
    std::is_same<T, BTFPowersD<typename T::value_type>>::value>>
{
    void operator()(T const& stream_data, 
                    ObservationHeader const& obs_header,
                    PipelineConfig const& config,
                    Header& output_header){
        output_header.set<std::size_t>("NCHAN", stream_data.nchannels());
        output_header.set<std::size_t>("NSAMP", stream_data.nsamples());
        output_header.set<std::size_t>("NBEAM", stream_data.nbeams());
        output_header.set<long double>("COHERENT_DM",
                static_cast<long double>(stream_data.reference_dm()));
        output_header.set<std::string>("STOKES_MODE", config.stokes_mode());
        output_header.set<std::size_t>("NPOL", config.stokes_mode().size());
        output_header.set<std::size_t>("NDIM", 1);
        output_header.set<std::size_t>("NBIT", sizeof(typename T::value_type) * 8);
        output_header.set<std::string>("DTYPE", "INTENSITIES");
    }
};

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
FileOutputStream&
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

    _file_streams[stream_idx] = std::make_unique<FileOutputStream>(
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
            psrdada_cpp::RawBytes bytes(temp_header,
                                        header_size,
                                        header_size,
                                        false);
            Header header_writer(bytes);

            // Default boilerplate keys
            header_writer.set<std::string>("HEADER", "DADA");
            header_writer.set<std::string>("HDR_VERSION", "1.0");
            header_writer.set<std::size_t>("HDR_SIZE", 4096);
            header_writer.set<std::string>("DADA_VERSION", "1.0");

            // Specific pointing and telescope info
            header_writer.set<std::string>("SOURCE", _header.source_name);
            header_writer.set<std::string>("RA", _header.ra);
            header_writer.set<std::string>("DEC", _header.dec);
            header_writer.set<std::string>("TELESCOPE", _header.telescope);
            header_writer.set<std::string>("INSTRUMENT", _header.instrument);

            // Params describing the whole observation
            header_writer.set<std::size_t>("OBS_NCHAN", _header.obs_nchans);
            header_writer.set<long double>("OBS_FREQUENCY",
                                           _header.obs_frequency);
            header_writer.set<long double>("OBS_BW", _header.obs_bandwidth);

            // General params specific to this processing
            header_writer.set<long double>("FREQ", _header.frequency);
            header_writer.set<long double>("BW", _header.bandwidth);
            header_writer.set<long double>("TSAMP", stream_data.tsamp() * 1e6);
            header_writer.set<std::size_t>("CHAN0_IDX", _header.chan0_idx);
            header_writer.set<long double>("UTC_START",
                                           _header.utc_start +
                                               stream_data.utc_offset());
            header_writer.set<std::string>("ORDER", stream_data.dims_as_string());

            // File writing management params
            header_writer.set<std::size_t>("FILE_SIZE", filesize);
            header_writer.set<std::size_t>("FILE_NUMBER", file_idx);
            header_writer.set<std::size_t>("OBS_OFFSET", bytes_written);
            header_writer.set<std::size_t>("OBS_OVERLAP", 0);  

            // Below we make a callback to type specific helpers
            // this is done as the handling of things like NPOL,
            // NDIM, STOKES_MODE, etc. is type specific. 
            HeaderFormatter<VectorType>()(stream_data, _header, _config, header_writer);

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
               << stream_idx << "/" 
               << std::fixed << std::setfill('0') << std::setw(9)
               << static_cast<int>(_header.frequency);
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
    base_filename << get_formatted_time(_header.utc_start) << "_" << stream_idx
                  << "_" << std::fixed << std::setprecision(3)
                  << std::setfill('0') << std::setw(9)
                  << stream_data.reference_dm() << "_" << std::setprecision(0)
                  << std::setfill('0') << std::setw(9) << _header.frequency;
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
            ->write(reinterpret_cast<char const*>(
                        thrust::raw_pointer_cast(stream_data.data())),
                    stream_data.size() *
                        sizeof(typename VectorType::value_type));
    }
    return false;
}

} // namespace skyweaver