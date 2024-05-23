#ifndef SKYWEAVER_DEDISPERSER_COHERENTDEDISPERSER_HPP
#define SKYWEAVER_DEDISPERSER_COHERENTDEDISPERSER_HPP

#include "cufft.h"
#include "skyweaver/PipelineConfig.hpp"
#include "skyweaver/skyweaver_constants.hpp"

#include <cufft.h>
#include <psrdada_cpp/psrdadaheader.hpp>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <vector>
#define DM_CONST 4.148806423e9 // MHz^2 pc^-1 cm^3 us
#define PI       acos(-1.0)
#define TWO_PI   2 * acos(-1.0)

namespace skyweaver
{
namespace kernels
{

__global__ void dedisperse(cufftComplex const* __restrict__ _d_ism_response,
                           cufftComplex const* in,
                           cufftComplex* out,
                           unsigned total_chans);

}

struct CoherentDedisperserConfig {
    std::size_t num_fine_chans; // i.e FFT Length
    std::size_t num_coarse_chans;
    std::size_t num_pols;
    std::size_t num_antennas;
    float tsamp;
    float bw;
    float low_freq;
    float high_freq;
    float coarse_chan_bw;

    float fine_chan_bw;

    thrust::host_vector<float> _h_dms;
    thrust::device_vector<float> _d_dms;
    thrust::device_vector<float> _d_dm_prefactor;
    // thrust::device_vector<cufftComplex> _d_ism_responses; // flattened array
    // of ISM responses following the order of DMs, channels and sub-channels
    std::vector<thrust::device_vector<cufftComplex>> _d_ism_responses;

    cufftHandle _fft_plan;
    // cufftHandle _i_fft_plan;
};
void get_dm_responses(CoherentDedisperserConfig& config,
                      float dm_prefactor,
                      thrust::device_vector<cufftComplex>& responses);

class CoherentDedisperser
{
  public:
    static void createConfig(CoherentDedisperserConfig& config,
                             std::size_t num_fine_chans,
                             std::size_t num_coarse_chans,
                             std::size_t num_pols,
                             std::size_t num_antennas,
                             float tsamp,
                             float low_freq,
                             float bw,
                             std::vector<float> dms);
    CoherentDedisperser(PipelineConfig const& pipeline_config,
                        CoherentDedisperserConfig& config)
        : pipeline_config(pipeline_config), config(config)
    {
    }
    ~CoherentDedisperser(){};
    void dedisperse(thrust::device_vector<char2> const& d_tpa_voltages_in,
                    thrust::device_vector<char2>& d_ftpa_voltages_out,
                    std::size_t out_offset,
                    int dm_idx);

  private:
    PipelineConfig const& pipeline_config;
    CoherentDedisperserConfig& config;
    thrust::device_vector<cufftComplex> d_fpa_spectra;
    thrust::device_vector<cufftComplex> d_tpa_voltages_out_temp;
    thrust::device_vector<cufftComplex> d_tpa_voltages_temp;
    void multiply_by_chirp(
        thrust::device_vector<cufftComplex> const& d_fpa_spectra_in,
        thrust::device_vector<cufftComplex>& d_fpa_spectra_out,
        int dm_idx);
};
} // namespace skyweaver
#endif // DEDISPERSER_HPP
