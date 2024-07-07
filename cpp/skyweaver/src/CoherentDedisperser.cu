#include "cuda.h"
#include "cufft.h"
#include "skyweaver/CoherentDedisperser.cuh"

#include <cmath>
#include <cuda_runtime.h>
#include <iomanip>
#include <iostream>
#include <thrust/complex.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>
#include <vector>
namespace skyweaver
{

double CoherentDedisperser::get_dm_delay(double f1, double f2, double dm)
{
    return ((1 / pow(f1 / 1000, 2.0)) - (1 / pow(f2 / 1000, 2.0))) * dm *
           0.00415;
}
void CoherentDedisperser::createConfig(CoherentDedisperserConfig& config,
                                       std::size_t fft_length,
                                       std::size_t overlap_samps,
                                       std::size_t num_coarse_chans,
                                       std::size_t npols,
                                       std::size_t nantennas,
                                       double tsamp,
                                       double low_freq,
                                       double bw,
                                       std::vector<float> dms)
{
    config.fft_length       = fft_length;
    config.overlap_samps    = overlap_samps;
    config.num_coarse_chans = num_coarse_chans;
    config.npols            = npols;
    config.nantennas        = nantennas;
    config.tsamp            = tsamp;

    config.low_freq       = low_freq;
    config.bw             = bw;
    config.high_freq      = low_freq + bw;
    config.coarse_chan_bw = bw / num_coarse_chans;

    /* Precompute DM constants */
    config._h_dms = dms;
    config._d_dms = config._h_dms;
    config._d_dm_prefactor.resize(dms.size());
    config._d_ism_responses.resize(dms.size());
    for(int i = 0; i < dms.size(); i++) {
        config._d_ism_responses[i].resize(num_coarse_chans * fft_length);
    }

    thrust::transform(
        config._d_dms.begin(),
        config._d_dms.end(),
        config._d_dm_prefactor.begin(),
        [=] __device__(double dm) { return -1.0f * TWO_PI * DM_CONST * dm; });

    config.fine_chan_bw = config.coarse_chan_bw / config.fft_length;

    for(int idx = 0; idx < config._d_dm_prefactor.size(); idx++) {
        get_dm_responses(config,
                         config._d_dm_prefactor[idx],
                         config._d_ism_responses[idx]);
    }

    // data is FTPA order, we will loop over F, so we are left with TPA order.
    // Let's fuse PA to X, so TX order.
    //  We stride and batch over X and transform T
    std::size_t X  = config.npols * config.nantennas;
    int n[1]       = {static_cast<int>(fft_length)}; // FFT size
    int inembed[1] = {static_cast<int>(fft_length)};
    int onembed[1] = {static_cast<int>(fft_length)};
    int istride    = X;
    int ostride    = X;
    int idist      = 1;
    int odist      = 1;
    int batch      = X;

    if(cufftPlanMany(&config._fft_plan,
                     1,
                     n,
                     inembed,
                     istride,
                     idist,
                     onembed,
                     ostride,
                     odist,
                     CUFFT_C2C,
                     batch) != CUFFT_SUCCESS) {
        std::runtime_error("CUFFT error: Plan creation failed");
    }

    BOOST_LOG_TRIVIAL(debug) << "FFT plan created";
}

/**
 **/
namespace
{
#define NCHANS_PER_BLOCK 128
} // namespace

void CoherentDedisperser::dedisperse(
    TPAVoltagesD<char2> const& d_tpa_voltages_in,
    FTPAVoltagesD<char2>& d_ftpa_voltages_out,
    unsigned int freq_idx,
    unsigned int dm_idx)
{
    d_fpa_spectra.resize(d_tpa_voltages_in.size(), {0.0f, 0.0f});
    d_tpa_voltages_temp.resize(d_tpa_voltages_in.size(), {0.0f, 0.0f});
    d_tpa_voltages_out_temp.resize(d_tpa_voltages_in.size(), {0.0f, 0.0f});

    BOOST_LOG_TRIVIAL(debug)
        << "Input TPA voltages to dedisperse, d_tpa_voltages_in.size(): "
        << d_tpa_voltages_in.size();
    BOOST_LOG_TRIVIAL(debug)
        << "Output FTPA voltages to write to, d_ftpa_voltages_out.size(): "
        << d_ftpa_voltages_out.size();

    thrust::transform(d_tpa_voltages_in.begin(),
                      d_tpa_voltages_in.end(),
                      d_tpa_voltages_temp.begin(),
                      [=] __device__(char2 const& val) {
                          cufftComplex complex_val;
                          complex_val.x = val.x;
                          complex_val.y = val.y;
                          return complex_val;
                      });

    BOOST_LOG_TRIVIAL(debug) << "Transformed voltages to cufftComplex";

    cufftExecC2C(config._fft_plan,
                 thrust::raw_pointer_cast(d_tpa_voltages_temp.data()),
                 thrust::raw_pointer_cast(d_fpa_spectra.data()),
                 CUFFT_FORWARD);

    BOOST_LOG_TRIVIAL(debug) << "Executed forward FFT";

    multiply_by_chirp(d_fpa_spectra,
                      d_fpa_spectra,
                      freq_idx,
                      dm_idx); // operating in place..

    BOOST_LOG_TRIVIAL(debug) << "Multiplied by chirp";

    cufftExecC2C(config._fft_plan,
                 thrust::raw_pointer_cast(d_fpa_spectra.data()),
                 thrust::raw_pointer_cast(d_tpa_voltages_out_temp.data()),
                 CUFFT_INVERSE);

    BOOST_LOG_TRIVIAL(debug) << "Executed inverse FFT";

    std::size_t out_offset = freq_idx * config.nantennas * config.npols *
                             (config.fft_length - config.overlap_samps);
    std::size_t discard_size =
        config.nantennas * config.npols * config.overlap_samps / 2;

    BOOST_LOG_TRIVIAL(debug) << "Output offset to write from: " << out_offset;
    BOOST_LOG_TRIVIAL(debug) << "discard_size: " << discard_size;
    BOOST_LOG_TRIVIAL(debug)
        << "copying from input from " << discard_size << " to "
        << d_tpa_voltages_out_temp.size() - discard_size;
    BOOST_LOG_TRIVIAL(debug)
        << "copying to output from " << out_offset << " to "
        << out_offset + d_tpa_voltages_out_temp.size() - 2 * discard_size;

    std::size_t N = config.fft_length;

    // transform: divide by d_tpa_voltages_in.size()
    thrust::transform(d_tpa_voltages_out_temp.begin() + discard_size,
                      d_tpa_voltages_out_temp.end() - discard_size,
                      d_ftpa_voltages_out.begin() + out_offset,
                      [=] __device__(cufftComplex const& val) {
                          char2 char2_val;
                          char2_val.x = static_cast<char>(
                              __float2int_rn(val.x / N)); // scale the data back
                          char2_val.y =
                              static_cast<char>(__float2int_rn(val.y / N));
                          return char2_val;
                      });

    BOOST_LOG_TRIVIAL(debug) << "Transformed cufftComplex to char2";
    BOOST_LOG_TRIVIAL(info) << "Copied data, done with this dedispersion call";
}

void CoherentDedisperser::multiply_by_chirp(

    thrust::device_vector<cufftComplex> const& d_fpa_spectra_in,
    thrust::device_vector<cufftComplex>& d_fpa_spectra_out,
    unsigned int freq_idx,
    unsigned int dm_idx)
{
    std::size_t total_chans     = config._d_ism_responses[dm_idx].size();
    std::size_t response_offset = freq_idx * config.fft_length;

    BOOST_LOG_TRIVIAL(debug) << "Freq idx: " << freq_idx;
    BOOST_LOG_TRIVIAL(debug) << "response_offset: " << response_offset;

    dim3 blockSize(config.nantennas * config.npols);
    dim3 gridSize(config.fft_length / NCHANS_PER_BLOCK);
    kernels::dedisperse<<<gridSize, blockSize>>>(
        thrust::raw_pointer_cast(config._d_ism_responses[dm_idx].data() +
                                 response_offset),
        thrust::raw_pointer_cast(d_fpa_spectra_in.data()),
        thrust::raw_pointer_cast(d_fpa_spectra_out.data()),
        total_chans);
}
} // namespace skyweaver
namespace skyweaver
{
namespace kernels
{

__global__ void dedisperse(cufftComplex const* __restrict__ _d_ism_response,
                           cufftComplex const* in,
                           cufftComplex* out,
                           unsigned total_chans)
{
    const unsigned pa_size = blockDim.x;

    volatile __shared__ cufftComplex response[NCHANS_PER_BLOCK];

    const unsigned block_start_chan_idx = blockIdx.x * NCHANS_PER_BLOCK;

    const unsigned remainder =
        min(total_chans - block_start_chan_idx, NCHANS_PER_BLOCK);

    for(int idx = threadIdx.x; idx < remainder; idx += pa_size) {
        cufftComplex const temp = _d_ism_response[block_start_chan_idx + idx];
        response[idx].x         = temp.x;
        response[idx].y         = temp.y;
    }

    __syncthreads();

    /**
    Each block processes NANT * NPOL in parallel (one per thread)
    Each thread processes NCHANS_PER_BLOCK channels sequentially, for a given
    (iant, ipol)
    **/

    for(int block_ichan_idx = 0; block_ichan_idx < remainder;
        ++block_ichan_idx) {
        const int chan_idx = (block_start_chan_idx + block_ichan_idx) *
                                 pa_size    // get to the correct chan_idx
                             + threadIdx.x; // get to the correct (iant, ipol)
        out[chan_idx] = cuCmulf(response[block_ichan_idx], in[chan_idx]);
    }
}

struct DMResponse {
    int num_coarse_chans;
    double low_freq;
    double coarse_chan_bw;
    double fine_chan_bw;
    double dm_prefactor;
    int num_fine_chans;

    DMResponse(int num_coarse_chans,
               int num_fine_chans,
               double low_freq,
               double coarse_chan_bw,
               double fine_chan_bw,
               double dm_prefactor)
        : num_coarse_chans(num_coarse_chans), num_fine_chans(num_fine_chans),
          low_freq(low_freq), coarse_chan_bw(coarse_chan_bw),
          fine_chan_bw(fine_chan_bw), dm_prefactor(dm_prefactor)
    {
    }

    __device__ inline cufftComplex operator()(int tid) const
    {
        int chan      = tid / num_fine_chans; // Coarse channel
        int fine_chan = tid % num_fine_chans; // fine channel

        double nu_0 = low_freq + chan * coarse_chan_bw -
                      0.5f * coarse_chan_bw; // + fine_chan * fine_chan_bw;

        double nu = fine_chan * fine_chan_bw; // fine_chan_freq

        double phase_prefactor = nu * nu * dm_prefactor;
        double phase =
            phase_prefactor / ((nu_0 + nu) * nu_0 * nu_0); // precalculate
        cufftDoubleComplex weight;
        sincos(phase,
               &weight.y,
               &weight.x); // TO DO: test if it is not approximate
        cufftComplex float_weight;
        float_weight.x = static_cast<float>(weight.x);
        float_weight.y = static_cast<float>(weight.y);

        return float_weight;
    }
};

} // namespace kernels

void get_dm_responses(CoherentDedisperserConfig& config,
                      double dm_prefactor,
                      thrust::device_vector<cufftComplex>& response)
{
    BOOST_LOG_TRIVIAL(info) << "Generating DM responses";
    thrust::device_vector<int> indices(config.num_coarse_chans *
                                       config.fft_length);
    thrust::sequence(indices.begin(), indices.end());

    // Apply the DMResponse functor using thrust's transform
    thrust::transform(indices.begin(),
                      indices.end(),
                      response.begin(),
                      kernels::DMResponse(config.num_coarse_chans,
                                          config.fft_length,
                                          config.low_freq,
                                          config.coarse_chan_bw,
                                          config.fine_chan_bw,
                                          dm_prefactor));
}

} // namespace skyweaver
