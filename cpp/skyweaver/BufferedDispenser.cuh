#ifndef BUFFERED_DISPENSER_HPP
#define BUFFERED_DISPENSER_HPP

#include "cuda.h"
#include "skyweaver/DescribedVector.hpp"
#include "skyweaver/MultiFileReader.cuh"
#include "skyweaver/PipelineConfig.hpp"
#include "skyweaver/CoherentDedisperser.cuh"


#include <cassert>
#include <memory>
#include <thrust/complex.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>

//TODO: Document this interface

namespace skyweaver
{
class BufferedDispenser;
}

class skyweaver::BufferedDispenser
{
  public:
    typedef FTPAVoltagesH<char2> FTPAVoltagesH;
    typedef FTPAVoltagesD<char2> FTPAVoltagesTypeD;
    typedef TPAVoltagesH<char2> HostTPAVoltagesH;
    typedef TPAVoltagesD<char2> TPAVoltagesTypeD;

  private:
    PipelineConfig const& _config;
    std::size_t _block_length_tpa; // length of nsamples * nantennas * npol  but
                                   // not char2, per block of data
    std::size_t _max_delay_tpa;    // length of dedispersion kernel in samples *
                                   // nantennas * npol  but not char2
    std::vector<TPAVoltagesTypeD>
        _d_prev_channeled_tpa_voltages; // stores it until next iteration. This
                                        // is a buffer of kernel length size for
                                        // all channels in FTPA order
    std::vector<TPAVoltagesTypeD>
        _d_channeled_tpa_voltages; // NCHANS=64 * TPA vectors

    std::vector<bool> _first_hoard; // flag to know if we are hoarding for the
                                    // first time
    
    cudaStream_t _stream;

  public:
    BufferedDispenser(PipelineConfig const& config, CoherentDedisperserConfig const& dedisp_config,  cudaStream_t stream);
    void hoard(FTPAVoltagesTypeD const& ftpa_voltages_in);
    TPAVoltagesTypeD const& dispense(std::size_t chan_idx) const;
};

#endif // BUFFERED_DISPENSER_HPP
