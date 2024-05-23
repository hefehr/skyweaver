#ifndef MULTIFILE_READER_HPP
#define MULTIFILE_READER_HPP

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <mutex>
#include <memory>
#include "psrdada_cpp/psrdadaheader.hpp"
#include "psrdada_cpp/raw_bytes.hpp"
#include "skyweaver/PipelineConfig.hpp"
#include <thrust/device_vector.h>
#include "cuda.h"
#include  <thrust/host_vector.h>
#include <cassert>
#include "skyweaver/ObservationHeader.hpp"
#include <memory>
namespace skyweaver
{
    class MultiFileReader;
}

class skyweaver::MultiFileReader
{

public:
    typedef thrust::device_vector<char2> VoltageType;

private:
    PipelineConfig const &_config;
    std::vector<std::string> files;
    std::vector<std::size_t> sizes;
    std::ifstream _current_stream;
    int _current_file_idx;
    std::size_t _current_position;
    std::mutex mtx;
    bool eofFlag;
    std::size_t _dada_header_size;
    std::size_t _total_size;
    bool _is_open;

    void read_header();
    std::unique_ptr<ObservationHeader> header;

public:
    MultiFileReader(PipelineConfig const &config);
    void open();
    void open_next();
    void open_previous();
    void close();
    void seekg(std::size_t pos, std::ios_base::seekdir dir = std::ios_base::beg);
    std::size_t tellg() const;
    bool eof() const;
    bool good() const;
    bool can_read(std::size_t bytes);
    bool has_next();
    void next();
    std::streamsize read(thrust::host_vector<char2>& buffer, std::streamsize bytes);
    bool is_open() const;

    template <typename T>
    friend MultiFileReader &operator>>(MultiFileReader &reader, T &value);
};

#endif // MULTIFILE_READER_HPP
