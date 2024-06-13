#include "psrdada_cpp/cuda_utils.hpp"
#include "skyweaver/DelayManager.cuh"
#include "thrust/copy.h"

#include <cerrno>
#include <cstring>
#include <errno.h>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>

namespace skyweaver
{

DelayManager::DelayManager(PipelineConfig const& config, cudaStream_t stream)
    : _config(config), _copy_stream(stream), _valid_nbeams(0),
      _valid_nantennas(0), _valid_nbeamsets(0)
{
    BOOST_LOG_TRIVIAL(debug) << "Constructing new DelayManager instance";
    std::string delay_file = _config.delay_file();
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

    // parse_nbeamsets is only being called on the first model
    // read from file. The operation is relatively high cost and
    // only needs recomputed if the beamsets change during a run.
    // Changing beamsets are thus not supported.
    _valid_nbeamsets = parse_beamsets();
    _valid_nantennas = _header.nantennas;
    _valid_nbeams    = _header.nbeams;
}

DelayManager::~DelayManager()
{
    if(_input_stream.is_open()) {
        _input_stream.close();
    }
}

double DelayManager::epoch() const
{
    return _header.start_epoch;
}

DelayManager::DelayVectorDType const& DelayManager::delays(double epoch)
{
    // This function should return the delays in GPU memory

    // Scan through the model file until we reach model that
    // contains valid delays for the given epoch or until we
    // hit EOF (which throws an exception).
    if(epoch < _header.start_epoch) {
        throw InvalidDelayEpoch(epoch);
    }

    while(!validate_model(epoch)) { read_next_model(); }
    if(_header.nantennas > _config.nantennas()) {
        throw std::runtime_error("Delay model contains too many antennas");
    }
    if(_header.nbeams > _config.nbeams()) {
        throw std::runtime_error("Delay model contains too many beams");
    }
    // Resize the arrays for the delay model on the GPU
    const std::size_t out_nelements = _config.nantennas() * _config.nbeams();
    _delays_d.resize(out_nelements, {0.0f, 0.0f, 0.0f});

    // In order to ensure that the correct sizes are available for the pipeline
    // implementation, we here pad the delays array out to the maximum number of
    // antennas and beams that will be processed by the pipeline.
    void* host_ptr =
        static_cast<void*>(thrust::raw_pointer_cast(_delays_h.data()));
    void* dev_ptr =
        static_cast<void*>(thrust::raw_pointer_cast(_delays_d.data()));
    unsigned host_pitch = _header.nantennas * sizeof(DelayModel);
    unsigned dev_pitch  = _config.nantennas() * sizeof(DelayModel);
    unsigned ncols      = _header.nantennas * sizeof(DelayModel);
    unsigned nrows      = _header.nbeams;

    // This could be made async
    CUDA_ERROR_CHECK(cudaMemcpy2D(dev_ptr,
                                  dev_pitch,
                                  host_ptr,
                                  host_pitch,
                                  ncols,
                                  nrows,
                                  cudaMemcpyHostToDevice));
    return _delays_d;
}

bool DelayManager::validate_model(double epoch) const
{
    if((_header.nbeams != _valid_nbeams) ||
       (_header.nantennas != _valid_nantennas)) {
        throw std::runtime_error(
            "Variable delay model parameters are unsupported");
    }
    return ((epoch >= _header.start_epoch) && (epoch <= _header.end_epoch));
}

void DelayManager::safe_read(char* buffer, std::size_t nbytes)
{
    BOOST_LOG_TRIVIAL(debug)
        << "At byte " << _input_stream.tellg() << " of the input file";
    BOOST_LOG_TRIVIAL(debug) << "Safe reading " << nbytes << " bytes";
    _input_stream.read(buffer, nbytes);
    if(_input_stream.eof()) {
        // Reached the end of the delay model file
        // TODO: Decide what the behaviour should be here.
        throw std::runtime_error("Reached end of delay model file");
    } else if(_input_stream.fail() || _input_stream.bad()) {
        std::ostringstream error_msg;
        error_msg << "Error: Unable to read from delay file: "
                  << std::strerror(errno);
        throw std::runtime_error(error_msg.str().c_str());
    }
    BOOST_LOG_TRIVIAL(debug) << "Read complete";
}

void DelayManager::read_next_model()
{
    BOOST_LOG_TRIVIAL(debug) << "Reading delay model from file";
    // Read the model header
    safe_read(reinterpret_cast<char*>(&_header), sizeof(_header));

    BOOST_LOG_TRIVIAL(debug) << "Delay model read successful";
    BOOST_LOG_TRIVIAL(debug)
        << "Delay model parameters: " << "Nantennas = " << _header.nantennas
        << ", " << "Nbeams = " << _header.nbeams << ", "
        << "Start = " << _header.start_epoch << ", "
        << "End = " << _header.end_epoch;

    const std::size_t nelements = _header.nantennas * _header.nbeams;
    _delays_h.resize(nelements);
    // Read the weight, offset, rate tuples from the file
    BOOST_LOG_TRIVIAL(debug) << "buffer size: " << _delays_h.size() * sizeof(decltype(_delays_h)::value_type);
    safe_read(
        reinterpret_cast<char*>(thrust::raw_pointer_cast(_delays_h.data())),
        nelements * sizeof(DelayVectorHType::value_type));
}

std::size_t DelayManager::parse_beamsets()
{
    // Note there is a strong assumption here that
    // all beams in a beamset are contiguous
    BOOST_LOG_TRIVIAL(debug) << "Parsing beamsets";
    std::vector<thrust::host_vector<float>> beamsets_weights;
    thrust::host_vector<int> beamset_map(_header.nbeams);
    beamsets_weights.emplace_back(_header.nantennas);

    int beamset_idx     = 0;
    bool beamset_update = false;
    // The first beam always belongs to the first beamset
    beamset_map[0] = 0;
    // Populate the weights for the first beamset
    for(int ant_idx = 0; ant_idx < _header.nantennas; ++ant_idx) {
        beamsets_weights[beamset_idx][ant_idx] = _delays_h[ant_idx].x;
    }
    // Create a space for the next beamset (may not be required)
    beamsets_weights.emplace_back(_header.nantennas);

    // Start on the next beamset (which may or may not exist)
    ++beamset_idx;

    // Start from the 2nd beam in the set
    for(int beam_idx = 1; beam_idx < _header.nbeams; ++beam_idx) {
        for(int ant_idx = 0; ant_idx < _header.nantennas; ++ant_idx) {
            // Populate the values for the next beamset
            beamsets_weights[beamset_idx][ant_idx] =
                    _delays_h[beam_idx * _header.nantennas + ant_idx].x;

            // Check if the current weight is different from the previous weight
            // If we already know there is an update this check can be skipped
            if(!beamset_update &&
               (beamsets_weights[beamset_idx][ant_idx] !=
                beamsets_weights[beamset_idx - 1][ant_idx])) {
                // Mark that we are now on a new beamset
                beamset_update = true;
            }
        }
        if(beamset_update) {
            // If the last beam belonged to a new beamset then
            // update all arrays and specify the beamset in the
            // mapping array
            beamset_update        = false;
            beamset_map[beam_idx] = beamset_idx;
            beamsets_weights.emplace_back(_header.nantennas);
            ++beamset_idx;
        } else {
            beamset_map[beam_idx] = beamset_idx - 1;
        }
    }
    // There is no circumstance in which the last entry
    // in the beamsets array is needed. It is either
    // emtpy if the last beam belonged to its own beamset
    // or it is a copy of the last valid beamset.
    beamsets_weights.pop_back();
    BOOST_LOG_TRIVIAL(debug) << "Found " << beamsets_weights.size() << " beamsets";
    // At this point we can copy the weights
    // and mappings to the GPU
    _beamset_map_d = beamset_map;
    _weights_d.resize(beamsets_weights.size() * _header.nantennas);
    for(int ii = 0; ii < beamsets_weights.size(); ++ii) {  
        thrust::copy(beamsets_weights[ii].begin(),
                     beamsets_weights[ii].end(),
                     _weights_d.begin()) + ii * _header.nantennas;
    }
    // Return the number of beamsets
    return beamsets_weights.size();
}

DelayManager::BeamsetWeightsVectorType const& DelayManager::beamset_weights() const
{
    return _weights_d;
}

DelayManager::BeamsetMappingVectorType const& DelayManager::beamset_mapping() const
{
    return _beamset_map_d;
}

int DelayManager::nbeamsets() const
{
    return _valid_nbeamsets;
}

} // namespace skyweaver