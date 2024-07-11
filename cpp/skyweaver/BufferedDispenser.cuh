#ifndef BUFFERED_DISPENSER_HPP
#define BUFFERED_DISPENSER_HPP

#pragma once
#include "cuda.h"
#include "skyweaver/DescribedVector.hpp"
#include "skyweaver/MultiFileReader.cuh"
#include "skyweaver/PipelineConfig.hpp"

#include <cassert>
#include <memory>
#include <thrust/complex.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>

namespace skyweaver
{
class BufferedDispenser;
}

class skyweaver::BufferedDispenser
{
  public:
    typedef FTPAVoltagesH<char2> HostFTPAVoltagesType;
    typedef FTPAVoltagesD<char2> DeviceFTPAVoltagesType;
    typedef TPAVoltagesH<char2> HostTPAVoltagesType;
    typedef TPAVoltagesD<char2> DeviceTPAVoltagesType;

  private:
    PipelineConfig const& _config;
    std::size_t _block_length_tpa; // length of nsamples * nantennas * npol  but
                                   // not char2, per block of data
    std::size_t _max_delay_tpa;    // length of dedispersion kernel in samples
                                   // * nantennas * npol  but not char2
    std::vector<DeviceTPAVoltagesType>
        _d_prev_channeled_tpa_voltages; // stores it until next iteration. This
                                        // is a buffer of kernel length size for
                                        // all channels in FTPA order
    std::vector<DeviceTPAVoltagesType>
        _d_channeled_tpa_voltages; // NCHANS=64 * TPA vectors
    cudaStream_t _stream;

  public:
    BufferedDispenser(PipelineConfig const& config, cudaStream_t stream);
    void hoard(DeviceFTPAVoltagesType const& ftpa_voltages_in);
    DeviceTPAVoltagesType const& dispense(std::size_t chan_idx) const;
};

#endif // BUFFERED_DISPENSER_HPP
