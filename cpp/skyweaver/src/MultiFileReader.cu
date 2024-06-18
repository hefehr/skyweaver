#include "skyweaver/MultiFileReader.cuh"

#include <boost/log/trivial.hpp>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <thrust/host_vector.h>
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
    std::vector<char> headerBytes(_dada_header_size);

    for(const auto& file: files) {
        std::ifstream f(file, std::ifstream::in | std::ifstream::binary);
        if(!f.is_open()) {
            throw std::runtime_error("Could not open file " + file);
        }
        f.read(headerBytes.data(), _dada_header_size);
        read_header(headerBytes);
        f.seekg(0, std::ios::end); // Move to the end of the file
        std::size_t size = f.tellg();
        _total_size += (size - _dada_header_size); // skip header
        sizes.push_back(size - _dada_header_size);
        BOOST_LOG_TRIVIAL(debug)
            << "File: " << file << " Binary size:" << size - _dada_header_size;
        f.close();
    }
    this->_total_size = _total_size;
    BOOST_LOG_TRIVIAL(debug) << "Total size of data: " << _total_size;

    if(config.check_input_contiguity()) {
        check_contiguity();
    }

    open(); // open the first file
}

void MultiFileReader::check_contiguity()
{
    for(size_t i = 0; i < headers.size() - 1; i++) {
        auto header1 = headers[i];
        auto header2 = headers[i + 1];
        std::size_t header1_nsamples =
            get_total_size() / (header1.nantennas * header1.nchans *
                                header1.npol * header1.nbits * 2 / 8);
        BOOST_LOG_TRIVIAL(debug)
            << "Header1 mjd_start: " << std::setprecision(15)
            << header1.mjd_start;
        BOOST_LOG_TRIVIAL(debug)
            << "Header2 mjd_start: " << std::setprecision(15)
            << header2.mjd_start;
        BOOST_LOG_TRIVIAL(debug) << "Header1 nsamples: " << header1_nsamples;
        float time1 = header1_nsamples * header1.tsamp / 86400.0;
        if(header2.mjd_start != header1.mjd_start + time1) {
            throw std::runtime_error(
                "These are not contiguous:" + files[i] + " and " +
                files[i + 1] + " -> " +
                std::to_string(header1.mjd_start + time1) +
                " != " + std::to_string(header2.mjd_start));
        }
    }
}

MultiFileReader::~MultiFileReader()
{
    this->close();
}

void MultiFileReader::read_header(std::vector<char>& headerBytes)
{
    psrdada_cpp::RawBytes block(headerBytes.data(),
                                _dada_header_size,
                                _dada_header_size,
                                false);

    ObservationHeader header;
    read_dada_header(block, header);
    headers.push_back(header);

    BOOST_LOG_TRIVIAL(debug) << "Header parameters: " << this->header;
}

skyweaver::ObservationHeader const& MultiFileReader::get_header() const
{
    return headers[_current_file_idx];
}

void MultiFileReader::open()
{
    BOOST_LOG_TRIVIAL(debug) << "Opening file: " << files[_current_file_idx]
                             << "of size: " << sizes[_current_file_idx];
    _current_stream.open(files[_current_file_idx],
                         std::ifstream::in | std::ifstream::binary);
    if(_current_stream.good()) {
        _current_stream.seekg(
            _dada_header_size,
            std::ios::beg); // Skip header for subsequent files
    } else {
        throw std::runtime_error("Could not open file " +
                                 files[_current_file_idx]);
    }
    _current_position = _dada_header_size;
    eofFlag           = false;
    BOOST_LOG_TRIVIAL(debug) << "Current file: " << files[_current_file_idx];
}

void MultiFileReader::open_next()
{
    BOOST_LOG_TRIVIAL(debug)
        << "Opening next file: " << files[_current_file_idx]
        << "of size: " << sizes[_current_file_idx];
    _current_file_idx++;
    if(_current_file_idx < files.size()) {
        _current_stream.close();
        open();
    } else {
        eofFlag = true;
        BOOST_LOG_TRIVIAL(debug) << "End of files reached";
    }
}

void MultiFileReader::open_previous()
{
    _current_file_idx--;
    if(_current_file_idx >= 0) {
        _current_stream.close();
        open();
    } else {
        eofFlag = true;
    }
}

std::size_t MultiFileReader::get_total_size() const
{
    return _total_size;
}

bool MultiFileReader::can_read(std::size_t bytes) const
{
    std::size_t remainingSize = _total_size - _current_position;
    return remainingSize >= bytes;
}

void MultiFileReader::seekg(long pos, std::ios_base::seekdir dir)
{
    BOOST_LOG_TRIVIAL(debug) << "Seeking to " << pos << " bytes";
    std::size_t targetPos = (dir == std::ios_base::cur)
                                ? _current_position + pos
                                : pos;       // current : begin;
    if(dir == std::ios_base::end && pos < 0) // end
        targetPos = this->_total_size + pos;

    if(targetPos > _total_size) {
        throw std::runtime_error("Invalid seek position: " +
                                 std::to_string(targetPos));
    }

    std::size_t cumulativeSize = 0;
    for(size_t i = 0; i < files.size(); i++) {
        cumulativeSize += sizes[i];
        if(targetPos < cumulativeSize) {
            std::size_t filePos =
                targetPos - cumulativeSize + sizes[i]; // position in file
            if(_current_file_idx != i) {
                _current_stream.close();
                _current_file_idx = i;
                open();
            }

            _current_stream.seekg(_dada_header_size + filePos, std::ios::beg);
            BOOST_LOG_TRIVIAL(debug)
                << "Actually seeking to " << filePos << " bytes from "
                << files[_current_file_idx];
            BOOST_LOG_TRIVIAL(debug)
                << "Current position: " << _current_stream.tellg();
            _current_position = targetPos;
            eofFlag           = false;
            break;
        } else {
            BOOST_LOG_TRIVIAL(debug) << "Seek is after file: " << files[i];
        }
    }
    if(_current_position >= _total_size || targetPos >= _total_size) {
        eofFlag = true;
    }
}

void MultiFileReader::close()
{
    if(_current_stream.is_open()) {
        _current_stream.close();
    }
    eofFlag = true;
}

std::streamsize MultiFileReader::read(char* raw_ptr, std::streamsize bytes)
{
    std::streamsize totalRead = 0;

    while(bytes > 0) {
        BOOST_LOG_TRIVIAL(debug)
            << "Attempting to read " << bytes << " bytes starting from "
            << _current_stream.tellg() << " of " << files[_current_file_idx];

        if(!_current_stream.is_open()) {
            std::stringstream ss;
            ss << "File not open: " << files[_current_file_idx];
            throw std::runtime_error(ss.str());
        }

        _current_stream.read(raw_ptr + totalRead, bytes);

        std::streamsize bytesRead = _current_stream.gcount();
        BOOST_LOG_TRIVIAL(debug) << "Read " << bytesRead << " bytes from "
                                 << files[_current_file_idx];

        if(bytesRead < bytes) {
            BOOST_LOG_TRIVIAL(debug)
                << "This is less than required, opening next file"
                << files[_current_file_idx];
            open_next();
        }

        bytes -= bytesRead;
        totalRead += bytesRead;
        _current_position += bytesRead;

        if(eofFlag)
            break;
    }
    if(_current_position >= _total_size) {
        eofFlag = true;
    }
    BOOST_LOG_TRIVIAL(debug) << "Total read: " << totalRead;
    BOOST_LOG_TRIVIAL(debug) << "Current position: " << _current_position;
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
    if(!(reader._current_stream >> value)) {
        reader.eofFlag = true;
    }
    return reader;
}
