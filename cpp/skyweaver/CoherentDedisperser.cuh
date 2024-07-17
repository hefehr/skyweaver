#ifndef SKYWEAVER_DEDISPERSER_COHERENTDEDISPERSER_HPP
#define SKYWEAVER_DEDISPERSER_COHERENTDEDISPERSER_HPP

#include "skyweaver/DescribedVector.hpp"
#include "skyweaver/skyweaver_constants.hpp"
#include "skyweaver/PipelineConfig.hpp"
#include <boost/log/trivial.hpp>
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
    std::size_t fft_length; // i.e gulp length, number of fine channels
    std::size_t overlap_samps;
    std::size_t num_coarse_chans;
    std::size_t npols;
    std::size_t nantennas;
    double tsamp;
    double bw;
    double low_freq;
    double high_freq;
    double coarse_chan_bw;
    double fine_chan_bw;

    thrust::host_vector<float> _h_dms;
    thrust::device_vector<float> _d_dms;
    thrust::device_vector<double> _d_dm_prefactor;
    // thrust::device_vector<cufftComplex> _d_ism_responses; // flattened array
    // of ISM responses following the order of DMs, channels and sub-channels
    std::vector<thrust::device_vector<cufftComplex>> _d_ism_responses;

    cufftHandle _fft_plan;
    // cufftHandle _i_fft_plan;
};

void get_dm_responses(CoherentDedisperserConfig& config,
                      double dm_prefactor,
                      thrust::device_vector<cufftComplex>& responses);

void create_coherent_dedisperser_config(CoherentDedisperserConfig& config,
                        PipelineConfig const& pipeline_config);

void create_coherent_dedisperser_config(CoherentDedisperserConfig& config,
                       std::size_t fft_length,
                       std::size_t overlap_samps,
                       std::size_t num_coarse_chans,
                       std::size_t npols,
                       std::size_t nantennas,
                       double tsamp,
                       double low_freq,
                       double bw,
                       std::vector<float> dms);
class CoherentDedisperser
{
  public:
    
    static double get_dm_delay(double f1, double f2, double dm); // f1 and f2 in MHz
    CoherentDedisperser(CoherentDedisperserConfig& config): config(config) {}
    ~CoherentDedisperser() {};
    void dedisperse(TPAVoltagesD<char2> const& d_tpa_voltages_in,
                    FTPAVoltagesD<char2>& d_ftpa_voltages_out,
                    unsigned int freq_idx,
                    unsigned int dm_idx);

  private:
    CoherentDedisperserConfig& config;
    thrust::device_vector<cufftComplex> d_fpa_spectra;
    thrust::device_vector<cufftComplex> d_tpa_voltages_out_temp;
    thrust::device_vector<cufftComplex> d_tpa_voltages_temp;
    void multiply_by_chirp(
        thrust::device_vector<cufftComplex> const& d_fpa_spectra_in,
        thrust::device_vector<cufftComplex>& d_fpa_spectra_out,
        unsigned int freq_idx,
        unsigned int dm_idx);
};
} // namespace skyweaver
#endif // DEDISPERSER_HPP
