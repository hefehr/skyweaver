#include "skyweaver/BufferedDispenser.cuh"
using namespace skyweaver;
BufferedDispenser::BufferedDispenser(PipelineConfig const& config, cudaStream_t stream)
    : _config(config), _stream(stream)
{
    this->_block_length_tpa =
        _config.nantennas() * _config.npol() * _config.gulp_length_samps();
    this->_max_delay_tpa = _config.dedisp_max_delay_samps() *
                               _config.nantennas() * _config.npol();
    
    // this->_d_prev_ftpa_voltages.resize(_nchans * _max_delay_tpa);

    _d_channeled_tpa_voltages.resize(_config.nchans());
    _d_prev_channeled_tpa_voltages.resize(_config.nchans());

    for(std::size_t i = 0; i < _config.nchans(); i++) {
        _d_channeled_tpa_voltages[i].resize(_block_length_tpa +
                                            _max_delay_tpa);
        _d_prev_channeled_tpa_voltages[i].resize(_max_delay_tpa);
    }
}

void BufferedDispenser::hoard(DeviceVoltageType const& new_ftpa_voltages_in)
{
    char2 zeros;
    zeros.x = 0;
    zeros.y = 0;
    for(std::size_t i = 0; i < _config.nchans(); i++) {
        if(_d_prev_channeled_tpa_voltages.size() ==
           0) { // if first time set overlaps as zeros
           BOOST_LOG_TRIVIAL(debug) << "BD -> Filling TPA voltages " << i 
                                    << " with zeros up to length " << _max_delay_tpa; 
            thrust::fill(_d_channeled_tpa_voltages[i].begin(),
                         _d_channeled_tpa_voltages[i].begin() +
                             _max_delay_tpa,
                         zeros);

        } else { // first add corresponding overlap to output
            BOOST_LOG_TRIVIAL(debug) << "BD -> Copying previous voltages";
            thrust::copy(_d_prev_channeled_tpa_voltages[i].begin(),
                         _d_prev_channeled_tpa_voltages[i].end(),
                         _d_channeled_tpa_voltages[i].begin());
        }
        // then add the input data
        BOOST_LOG_TRIVIAL(debug) << "BD -> Copying new voltages";
        thrust::copy(new_ftpa_voltages_in.begin() + i * _block_length_tpa,
                     new_ftpa_voltages_in.begin() + (i + 1) * _block_length_tpa,
                     _d_channeled_tpa_voltages[i].begin() + _max_delay_tpa);

        // update the overlap for the next hoard
        BOOST_LOG_TRIVIAL(debug) << "BD -> Updating overlap";
        thrust::copy(new_ftpa_voltages_in.begin() +
                         (i + 1) * _block_length_tpa - _max_delay_tpa,
                     new_ftpa_voltages_in.begin() + (i + 1) * _block_length_tpa,
                     _d_prev_channeled_tpa_voltages[i].begin());
    }
}

BufferedDispenser::DeviceVoltageType const&
BufferedDispenser::dispense(std::size_t chan_idx) const
{ // implements overlapped buffering of data

    return _d_channeled_tpa_voltages[chan_idx];
}

// void BufferedDispenser::dispense(std::size_t chan_idx, DeviceVoltageType&
// tpa_voltages_out) { // implements overlapped buffering of data

//     std::size_t offset = _block_length_tpa * chan_idx; // offset to the
//     channel index

//     // copy Kernel length size of previous data to the next buffer
//     thrust::copy(_d_prev_ftpa_voltages.begin() + offset,
//                     _d_prev_ftpa_voltages.begin() + offset +
//                     max_delay_tpa, tpa_voltages_out.begin()); // copy
//                     last  Kernel Length size of data to the next buffer

//     // copy current input data to the next buffer
//     thrust::copy(ftpa_voltages_in.begin(),
//                     ftpa_voltages_in.end(),
//                     tpa_voltages_out.begin()
//                         + max_delay_tpa); // from offset -> end

//     // copy the last Kernel length size of data to the previous buffer for
//     the next iteration thrust::copy(ftpa_voltages_in.end() -
//     max_delay_tpa,
//                  ftpa_voltages_in.end(),
//                  _d_prev_ftpa_voltages.begin() + this->_block_length_tpa *
//                  chan_idx); // copy the data to the previous data buffer.

// }
