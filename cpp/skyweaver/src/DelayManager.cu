#include "psrdada_cpp/cuda_utils.hpp"
#include "skyweaver/DelayManager.cuh"

#include <cerrno>
#include <cstring>
#include <errno.h>
#include <fstream>
#include <iostream>
#include <sstream>

namespace skyweaver
{

DelayManager::DelayManager(std::string delay_file, cudaStream_t stream)
    : _copy_stream(stream)
{
  BOOST_LOG_TRIVIAL(debug) << "Constructing new DelayManager instance";
  BOOST_LOG_TRIVIAL(debug) << "Opening delay model file: " << delay_file;
  _input_stream.open(delay_file, std::ios::in | std::ios::binary);
  if(!_input_stream.is_open()) {
    std::ostringstream error_msg;
    error_msg << "Error: Unable to open file delay file due to error: "
              << std::strerror(errno);
    throw std::runtime_error(error_msg.str().c_str());
  }
  BOOST_LOG_TRIVIAL(debug) << "Delay model file successfully opened";

  // Get first model from file
  read_next_model();
}

DelayManager::~DelayManager()
{
  if(_input_stream.is_open()) {
    _input_stream.close();
  }
}

DelayManager::DelayVectorDType const& DelayManager::delays(double epoch)
{
  // This function should return the delays in GPU memory

  // Scan through the model file until we reach model that
  // contains valid delays for the given epoch or until we
  // hit EOF (which throws an exception).
  if (epoch < _header.start_epoch)
  {
    throw InvalidDelayEpoch(epoch);
  }
  
  while(!validate_model(epoch)) { read_next_model(); }
  
  thrust::copy(_delays_h.begin(), _delays_h.end(), _delays_d.begin());
  return _delays_d;
}

bool DelayManager::validate_model(double epoch) const
{
  return ((epoch >= _header.start_epoch) && (epoch <= _header.end_epoch));
}

void DelayManager::safe_read(char* buffer, std::size_t nbytes)
{
  BOOST_LOG_TRIVIAL(debug) << "At byte " << _input_stream.tellg() << " of the input file";
  _input_stream.read(buffer, nbytes);
  if(_input_stream.eof()) {
    // Reached the end of the delay model file
    // TODO: Decide what the behaviour should be here.
    throw std::runtime_error("Reached end of delay model file");
  } else if (_input_stream.fail() || _input_stream.bad()) {
    std::ostringstream error_msg;
    error_msg << "Error: Unable to read from delay file: "
              << std::strerror(errno);
    throw std::runtime_error(error_msg.str().c_str());
  }
}

void DelayManager::read_next_model()
{
  BOOST_LOG_TRIVIAL(debug) << "Reading delay model from file";
  // Read the model header
  safe_read(reinterpret_cast<char*>(&_header), sizeof(_header));

  BOOST_LOG_TRIVIAL(debug) << "Delay model read successful";
  BOOST_LOG_TRIVIAL(debug) << "Delay model parameters: "
                           << "Nantennas = " << _header.nantennas << ", "
                           << "Nbeams = " << _header.nbeams << ", "
                           << "Start = " << _header.start_epoch << ", "
                           << "End = " << _header.end_epoch;

  // Resize the arrays for the delay model on the host and GPU
  const std::size_t nelements = _header.nantennas * _header.nbeams;
  _delays_h.resize(nelements);
  _delays_d.resize(nelements);

  // Read the weight, offset, rate tuples from the file
  safe_read(reinterpret_cast<char*>(thrust::raw_pointer_cast(_delays_h.data())),
            nelements * sizeof(DelayVectorHType::value_type));
}

} // namespace skyweaver