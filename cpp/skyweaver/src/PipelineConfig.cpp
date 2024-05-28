#include "skyweaver/PipelineConfig.hpp"

#include <boost/algorithm/string.hpp>
#include <fstream>

namespace skyweaver
{

PipelineConfig::PipelineConfig()
    : _delay_file("delays.swd"), _input_files({}),
      _check_input_contiguity(false), _dada_header_size(4096),
      _output_dir("./"), _max_output_filesize(10000000000000),
      _output_file_prefix(""), _statistics_file("./statistics.bin"),
      _coherent_dms({0.0f}), _dedisp_kernel_length_samps(8192),
      _cfreq(1284000000.0), _bw(13375000.0), _channel_frequencies_stale(true),
      _gulp_length_samps(4096), _total_nchans(4096), _output_level(24.0f),
      _cb_power_scaling(0.0f), _cb_power_offset(0.0f), _ib_power_scaling(0.0f),
      _ib_power_offset(0.0f)
{
}

PipelineConfig::~PipelineConfig()
{
}

void PipelineConfig::delay_file(std::string const& fpath)
{
    _delay_file = fpath;
}

std::string const& PipelineConfig::delay_file() const
{
    return _delay_file;
}

void PipelineConfig::input_files(std::vector<std::string> const& files)
{
    _input_files = files;
}

std::vector<std::string> const& PipelineConfig::input_files() const
{
    return _input_files;
}

void PipelineConfig::check_input_contiguity(bool check)
{
    _check_input_contiguity = check;
}

bool PipelineConfig::check_input_contiguity() const
{
    return _check_input_contiguity;
}

void PipelineConfig::dada_header_size(std::size_t size)
{
    _dada_header_size = size;
}

std::size_t PipelineConfig::dada_header_size() const
{
    return _dada_header_size;
}

void PipelineConfig::read_input_file_list(std::string filename)
{
    BOOST_LOG_TRIVIAL(debug) << "Reading input file list from " << filename;
    std::string line;
    std::ifstream ifs(filename.c_str());
    if(!ifs.is_open()) {
        std::cerr << "Unable to open input file list: " << filename << " ("
                  << std::strerror(errno) << ")\n";
        throw std::runtime_error(std::strerror(errno));
    }
    _input_files.resize(0);
    // TODO: Check that the files from the input list exist
    while(std::getline(ifs, line)) {
        BOOST_LOG_TRIVIAL(debug) << line;
        boost::algorithm::trim(line);
        if(line[0] == '#') {
            // Line is a comment
            BOOST_LOG_TRIVIAL(debug) << "is a comment";
            continue;
        } else {
            _input_files.push_back(line);
        }
        BOOST_LOG_TRIVIAL(debug) << "trimmed: " << line;
    }
    ifs.close();
}

void PipelineConfig::output_dir(std::string const& path)
{
    _output_dir = path;
}

std::string const& PipelineConfig::output_dir() const
{
    return _output_dir;
}

std::size_t PipelineConfig::max_output_filesize() const
{
    return _max_output_filesize;
}

void PipelineConfig::max_output_filesize(std::size_t nbytes)
{
    _max_output_filesize = nbytes;
}

std::string const& PipelineConfig::output_file_prefix() const
{
    return _output_file_prefix;
}

void PipelineConfig::output_file_prefix(std::string const& prefix)
{
    _output_file_prefix = prefix;
}

void PipelineConfig::statistics_file(std::string const& filename)
{
    _statistics_file = filename;
}

std::string const& PipelineConfig::statistics_file() const
{
    return _statistics_file;
}

double PipelineConfig::centre_frequency() const
{
    return _cfreq;
}

void PipelineConfig::centre_frequency(double cfreq)
{
    _cfreq                     = cfreq;
    _channel_frequencies_stale = true;
}

double PipelineConfig::bandwidth() const
{
    return _bw;
}

void PipelineConfig::bandwidth(double bw)
{
    _bw                        = bw;
    _channel_frequencies_stale = true;
}

std::vector<float> const& PipelineConfig::coherent_dms() const
{
    return _coherent_dms;
}

void PipelineConfig::coherent_dms(std::vector<float> const& coherent_dms)
{
    _coherent_dms = coherent_dms;
}

void PipelineConfig::dedisp_kernel_length_samps(std::size_t kernel_length)
{
    _dedisp_kernel_length_samps = kernel_length;
}

std::size_t PipelineConfig::dedisp_kernel_length_samps() const
{
    return _dedisp_kernel_length_samps;
}

std::vector<double> const& PipelineConfig::channel_frequencies() const
{
    if(_channel_frequencies_stale) {
        calculate_channel_frequencies();
    }
    return _channel_frequencies;
}

void PipelineConfig::calculate_channel_frequencies() const
{
    /**
     * Need to revisit this implementation as it is not clear how the
     * frequencies are labeled for the data out of the F-engine. Either
     * way is a roughly correct place-holder.
     */
    double chbw    = bandwidth() / nchans();
    double fbottom = centre_frequency() - bandwidth() / 2.0;
    _channel_frequencies.clear();
    for(std::size_t chan_idx = 0; chan_idx < nchans(); ++chan_idx) {
        _channel_frequencies.push_back(fbottom + chbw / 2.0 +
                                       (chbw * chan_idx));
    }
    _channel_frequencies_stale = false;
}

std::size_t PipelineConfig::gulp_length_samps() const
{
    return _gulp_length_samps;
}

void PipelineConfig::gulp_length_samps(std::size_t nsamps)
{
    _gulp_length_samps = nsamps;
}

void PipelineConfig::output_level(float level)
{
    _output_level = level;
}

float PipelineConfig::output_level() const
{
    return _output_level;
}

std::size_t PipelineConfig::total_nchans() const
{
    return _total_nchans;
}

void PipelineConfig::total_nchans(std::size_t nchans)
{
    _total_nchans = nchans;
}

} // namespace skyweaver
