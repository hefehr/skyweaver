#ifndef SKYCLEAVERCONFIG_HPP
#define SKYCLEAVERCONFIG_HPP
namespace skyweaver
{

class SkyCleaverConfig
{
    std::string _output_dir;
    std::string _root_dir;
    std::string _root_prefix;
    std::string _out_prefix;
    std::size_t _nthreads;
    std::size_t _nsamples_per_block;
    std::size_t _nchans;
    std::size_t _nbeams;
    std::size_t _max_ram_gb;
    std::size_t _max_output_filesize;
    std::size_t _stream_id;
    std::size_t _nbridges;
    std::size_t _ndms;
    std::string _stokes_mode;
    std::size_t _dada_header_size;

  public:
    SkyCleaverConfig()
        : _output_dir(""), _root_dir(""), _root_prefix(""), _out_prefix(""),
          _nthreads(0), _nsamples_per_block(0), _nchans(0), _nbeams(0),
          _max_ram_gb(0), _max_output_filesize(2147483647), _stream_id(0),
          _nbridges(64), _ndms(0), _stokes_mode("I"), _dada_header_size(4096)
    {
    }
    SkyCleaverConfig(SkyCleaverConfig const&) = delete;

    void output_dir(std::string output_dir) { _output_dir = output_dir; }
    void root_dir(std::string root_dir) { _root_dir = root_dir; }
    void root_prefix(std::string root_prefix) { _root_prefix = root_prefix; }
    void out_prefix(std::string out_prefix) { _out_prefix = out_prefix; }
    void nthreads(std::size_t nthreads) { _nthreads = nthreads; }
    void nsamples_per_block(std::size_t nsamples_per_block)
    {
        _nsamples_per_block = nsamples_per_block;
    }
    void nchans(std::size_t nchans) { _nchans = nchans; }
    void nbeams(std::size_t nbeams) { _nbeams = nbeams; }
    void max_ram_gb(std::size_t max_ram_gb) { _max_ram_gb = max_ram_gb; }
    void max_output_filesize(std::size_t max_output_filesize)
    {
        _max_output_filesize = max_output_filesize;
    }
    void stream_id(std::size_t stream_id) { _stream_id = stream_id; }
    void nbridges(std::size_t nbridges) { _nbridges = nbridges; }
    void ndms(std::size_t ndms) { _ndms = ndms; }
    void stokes_mode(std::string stokes_mode) { _stokes_mode = stokes_mode; }
    void dada_header_size(std::size_t dada_header_size)
    {
        _dada_header_size = dada_header_size;
    }

    std::string output_dir() const { return _output_dir; }
    std::string root_dir() const { return _root_dir; }
    std::string root_prefix() const { return _root_prefix; }
    std::string out_prefix() const { return _out_prefix; }
    std::size_t nthreads() const { return _nthreads; }
    std::size_t nsamples_per_block() const { return _nsamples_per_block; }
    std::size_t nchans() const { return _nchans; }
    std::size_t nbeams() const { return _nbeams; }
    std::size_t max_ram_gb() const { return _max_ram_gb; }
    std::size_t max_output_filesize() const { return _max_output_filesize; }
    std::size_t stream_id() const { return _stream_id; }
    std::size_t nbridges() const { return _nbridges; }
    std::size_t ndms() const { return _ndms; }
    std::string stokes_mode() const { return _stokes_mode; }
    std::size_t dada_header_size() const { return _dada_header_size; }
};
} // namespace skyweaver
#endif // SKYCLEAVERCONFIG_HPP