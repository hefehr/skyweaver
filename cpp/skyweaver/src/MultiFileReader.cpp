#include "skyweaver/MultiFileReader.hpp"

using namespace skyweaver;

MultiFileReader::MultiFileReader(PipelineConfig const& config)
    : _config(config), _current_file_idx(0), _current_position(0),
      eofFlag(false)
{
    static_assert(sizeof(char2) == 2 * sizeof(char),
                  "Size of char2 is not as expected");

    std::size_t _total_size = 0;
    this->files             = config.input_files();
    this->_dada_header_size = config.dada_header_size();
    for(const auto& file: files) {
        std::ifstream f(file, std::ifstream::ate | std::ifstream::binary);
        f.seekg(0, std::ios::end); // Move to the end of the file
        std::size_t size = f.tellg();
        _total_size += (size - _dada_header_size); // skip header
        sizes.push_back(size - _dada_header_size);
        f.close();
    }
    this->_total_size = _total_size;
    if(!files.empty()) {
        read_header(); // Read header from the first file to set
                       // _dada_header_size
    }
}

void MultiFileReader::read_header()
{
    std::ifstream f(files[0], std::ifstream::binary);
    std::vector<char> headerBytes(_dada_header_size);
    f.read(headerBytes.data(), _dada_header_size);

    psrdada_cpp::RawBytes block(headerBytes.data(),
                                _dada_header_size,
                                _dada_header_size,
                                false);

    this->header = std::make_unique<ObservationHeader>();
    read_dada_header(block, *header);

    f.close();
}

void MultiFileReader::open()
{
    std::lock_guard<std::mutex> lock(
        mtx); /* mutex released automatically at end of scope*/
    _current_stream.open(files[_current_file_idx], std::ifstream::binary);
    if(_dada_header_size > 0 && _current_stream.good()) {
        _current_stream.seekg(
            _dada_header_size,
            std::ios::beg); // Skip header for subsequent files
    }
    _current_position = _dada_header_size;
    eofFlag           = false;
}

void MultiFileReader::open_next()
{
    std::lock_guard<std::mutex> lock(mtx);
    _current_file_idx++;
    if(_current_file_idx < files.size()) {
        _current_stream.close();
        open();
    } else {
        eofFlag = true;
    }
}

void MultiFileReader::open_previous()
{
    std::lock_guard<std::mutex> lock(mtx);
    _current_file_idx--;
    if(_current_file_idx >= 0) {
        _current_stream.close();
        open();
    } else {
        eofFlag = true;
    }
}

bool MultiFileReader::can_read(std::size_t bytes)
{
    std::size_t remainingSize = _total_size - _current_position;
    return remainingSize >= bytes;
}

void MultiFileReader::seekg(std::size_t pos, std::ios_base::seekdir dir)
{
    std::lock_guard<std::mutex> lock(mtx);
    std::size_t targetPos =
        (dir == std::ios_base::cur) ? _current_position + pos : pos;
    if(dir == std::ios_base::end && pos < 0)
        targetPos = this->_total_size + pos;

    std::size_t cumulativeSize = 0;
    for(size_t i = 0; i < sizes.size(); i++) {
        cumulativeSize += sizes[i];
        if(targetPos < cumulativeSize) {
            std::size_t filePos =
                targetPos - (cumulativeSize - sizes[i]); // position in file
            if(_current_file_idx != i) {
                _current_stream.close();
                _current_file_idx = i;
                _current_stream.open(files[_current_file_idx],
                                     std::ifstream::binary);
            }

            _current_stream.seekg(_dada_header_size + filePos, std::ios::beg);
            _current_position = targetPos;
            eofFlag           = false;
            break;
        }
    }
}

void MultiFileReader::close()
{
    std::lock_guard<std::mutex> lock(mtx);
    if(_current_stream.is_open()) {
        _current_stream.close();
    }
    eofFlag = true;
}

std::streamsize MultiFileReader::read(thrust::host_vector<char2>& buffer,
                                      std::streamsize bytes)
{
    std::lock_guard<std::mutex> lock(mtx);
    std::streamsize totalRead = 0;
    // get raw pointer to the data
    char2* raw_ptr = thrust::raw_pointer_cast(buffer.data());
    while(bytes > 0) {
        _current_stream.read(reinterpret_cast<char*>(raw_ptr) + totalRead,
                             bytes);
        std::streamsize bytesRead = _current_stream.gcount();
        if(bytesRead == 0 || bytesRead < bytes) { // End of file or error
            open_next();
        } else {
            totalRead += bytesRead;
            bytes -= bytesRead;
            _current_position += bytesRead;

            if(eofFlag)
                break;
        }
    }
    return totalRead;
}

bool MultiFileReader::eof() const
{
    return eofFlag;
}

bool MultiFileReader::good() const
{
    return _current_stream.good() && !eofFlag;
}

std::size_t MultiFileReader::tellg() const
{
    return _current_position;
}

bool MultiFileReader::is_open() const
{
    return _current_stream.is_open();
}

template <typename T>
MultiFileReader& operator>>(MultiFileReader& reader, T& value)
{
    std::lock_guard<std::mutex> lock(reader.mtx);
    if(!(reader._current_stream >> value)) {
        reader.eofFlag = true;
    }
    return reader;
}
