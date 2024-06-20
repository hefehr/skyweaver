#include "cuda.h"
#include "cufft.h"
#include "skyweaver/CoherentDedisperser.cuh"

#include <cmath>
#include <cuda_runtime.h>
#include <iostream>
#include <thrust/complex.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>
#include <vector>
namespace skyweaver
{

float CoherentDedisperser::get_dm_delay(float f1, float f2, float dm)
{
    return ((1 / pow(f1 / 1000, 2.0)) - (1 / pow(f2 / 1000, 2.0))) * dm *
           0.00415;
}
void CoherentDedisperser::createConfig(CoherentDedisperserConfig& config,
                                       std::size_t fft_length,
                                       std::size_t overlap_samps,
                                       std::size_t num_coarse_chans,
                                       std::size_t num_pols,
                                       std::size_t num_antennas,
                                       float tsamp,
                                       float low_freq,
                                       float bw,
                                       std::vector<float> dms)
{
    config.fft_length       = fft_length;
    config.overlap_samps    = overlap_samps;
    config.num_coarse_chans = num_coarse_chans;
    config.num_pols         = num_pols;
    config.num_antennas     = num_antennas;
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
        [=] __device__(float dm) { return -1.0f * TWO_PI * DM_CONST * dm; });

    for(int idx = 0; idx < config._d_dm_prefactor.size(); idx++) {
        BOOST_LOG_TRIVIAL(info)
            << "DM Prefactor: " << config._d_dm_prefactor[idx];
    }

    config.fine_chan_bw = config.coarse_chan_bw / config.fft_length;

    for(int idx = 0; idx < config._d_dm_prefactor.size(); idx++) {
        get_dm_responses(config,
                         config._d_dm_prefactor[idx],
                         config._d_ism_responses[idx]);
    }

    // data is FTPA order, we will loop over F, so we are left with TPA order.
    // Let's fuse PA to X, so TX order.
    //  We stride and batch over X and transform T
    std::size_t X  = config.num_pols * config.num_antennas;
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
    thrust::device_vector<char2> const& d_tpa_voltages_in,
    thrust::device_vector<char2>& d_ftpa_voltages_out,
    std::size_t out_offset,
    int dm_idx)
{
    d_fpa_spectra.resize(d_tpa_voltages_in.size(), {0.0f, 0.0f});
    d_tpa_voltages_temp.resize(d_tpa_voltages_in.size(), {0.0f, 0.0f});
    d_tpa_voltages_out_temp.resize(d_tpa_voltages_in.size(), {0.0f, 0.0f});

    cudaError_t error;

    thrust::transform(d_tpa_voltages_in.begin(),
                      d_tpa_voltages_in.end(),
                      d_tpa_voltages_temp.begin(),
                      [=] __device__(char2 const& val) {
                          cufftComplex complex_val;
                          complex_val.x = val.x;
                          complex_val.y = val.y;
                          return complex_val;
                      });
    error = cudaGetLastError();
    if(error != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl;
    }
    BOOST_LOG_TRIVIAL(debug) << "Transformed voltages to cufftComplex";

    cufftExecC2C(config._fft_plan,
                 thrust::raw_pointer_cast(d_tpa_voltages_temp.data()),
                 thrust::raw_pointer_cast(d_fpa_spectra.data()),
                 CUFFT_FORWARD);

    BOOST_LOG_TRIVIAL(debug) << "Executed forward FFT";

    // for(int i=0; i < 10; i++) {
    //     cufftComplex x = d_fpa_spectra[i];
    //     BOOST_LOG_TRIVIAL(debug) << "d_fpa_spectra[" << i << "].x: " <<x.x;
    //     BOOST_LOG_TRIVIAL(debug) << "d_fpa_spectra[" << i << "].y: " << x.y;
    // }

    multiply_by_chirp(d_fpa_spectra,
                      d_fpa_spectra,
                      dm_idx); // operating in place.

    // for(int i=0; i < 10; i++) {
    //     cufftComplex x = d_fpa_spectra[i];
    //     BOOST_LOG_TRIVIAL(debug) << "after d_fpa_spectra[" << i << "].x: "
    //     <<x.x; BOOST_LOG_TRIVIAL(debug) << "after d_fpa_spectra[" << i <<
    //     "].y: " << x.y;
    // }

    BOOST_LOG_TRIVIAL(debug) << "Multiplied by chirp";
    error = cudaGetLastError();
    if(error != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl;
    }

    cufftExecC2C(config._fft_plan,
                 thrust::raw_pointer_cast(d_fpa_spectra.data()),
                 thrust::raw_pointer_cast(d_tpa_voltages_out_temp.data()),
                 CUFFT_INVERSE);

    error = cudaGetLastError();
    if(error != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl;
    }

    BOOST_LOG_TRIVIAL(debug) << "Executed inverse FFT";

    std::size_t N = config.fft_length;
    std::size_t discard_size =
        config.num_antennas * config.num_pols * config.overlap_samps / 2;
    BOOST_LOG_TRIVIAL(debug) << "copying from " << discard_size << " to "
                             << d_tpa_voltages_out_temp.size() - discard_size;
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
}

void CoherentDedisperser::multiply_by_chirp(

    thrust::device_vector<cufftComplex> const& d_fpa_spectra_in,
    thrust::device_vector<cufftComplex>& d_fpa_spectra_out,
    int dm_idx)
{
    std::size_t total_chans = config._d_ism_responses[dm_idx].size();
    std::size_t batchSize   = d_fpa_spectra_in.size() / total_chans;

    if(total_chans % NCHANS_PER_BLOCK != 0) {
        throw std::runtime_error(
            "Total chans need to be a multiple of NCHANS_PER_BLOCK");
    }
    BOOST_LOG_TRIVIAL(debug) << "Total chans: " << total_chans;
    BOOST_LOG_TRIVIAL(debug) << "Batch size: " << batchSize;
    BOOST_LOG_TRIVIAL(debug) << "config._d_ism_responses[dm_idx] size: "
                             << config._d_ism_responses[dm_idx].size();
    BOOST_LOG_TRIVIAL(debug)
        << "d_fpa_spectra_in size: " << d_fpa_spectra_in.size();
    BOOST_LOG_TRIVIAL(debug)
        << "d_fpa_spectra_out size: " << d_fpa_spectra_out.size();

    dim3 blockSize(pipeline_config.nantennas() *
                   pipeline_config.npol());        // 128 threads per block
    dim3 gridSize(total_chans / NCHANS_PER_BLOCK); // 4096 blocks
    kernels::dedisperse<<<gridSize, blockSize>>>(
        thrust::raw_pointer_cast(config._d_ism_responses[dm_idx].data()),
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
    /**
    _d_ism_response is the size of 64 * 8192 = 524288
    d_fpa_spectra_in size is (65536+8192) * 64 * 2 = 9437184
    d_fpa_spectra_out size is (65536+8192) * 64 * 2 = 9437184
    **/
    const unsigned tp_size = SKYWEAVER_NANTENNAS * SKYWEAVER_NPOL; // 128

    volatile __shared__ cufftComplex response[NCHANS_PER_BLOCK]; // 128

    const int start_chan_idx =
        blockIdx.x * NCHANS_PER_BLOCK; // 0->4096 * 128 = 0 -> 524288

    const int remainder = min(total_chans - start_chan_idx,
                              NCHANS_PER_BLOCK); // 524288 -> 0 or 128

    for(int idx = threadIdx.x; idx < remainder; idx += blockDim.x) {
        cufftComplex const temp = _d_ism_response[start_chan_idx + idx];
        response[idx].x         = temp.x;
        response[idx].y         = temp.y;
    }

    __syncthreads();

    for(int ii = 0; ii < remainder; ++ii) {
        const int chan_idx =
            ((ii + start_chan_idx) % NCHANS_PER_BLOCK) * tp_size + threadIdx.x;
        // out[chan_idx]      = cuCmulf(response[ii], in[chan_idx]);
        out[chan_idx] = cuCmulf(response[ii], in[chan_idx]);
    }
}

struct DMResponse {
    int num_coarse_chans;
    float low_freq;
    float coarse_chan_bw;
    float fine_chan_bw;
    float dmPrefix;
    float phase_prefactor;

    DMResponse(int num_coarse_chans,
               float low_freq,
               float coarse_chan_bw,
               float fine_chan_bw,
               float dmPrefix)
        : num_coarse_chans(num_coarse_chans), low_freq(low_freq),
          coarse_chan_bw(coarse_chan_bw), fine_chan_bw(fine_chan_bw),
          dmPrefix(dmPrefix),
          phase_prefactor(fine_chan_bw * fine_chan_bw * dmPrefix)
    {
    }

    __device__ inline cufftComplex operator()(int tid) const
    {
        int chan      = tid / num_coarse_chans; // Coarse channel
        int fine_chan = tid % num_coarse_chans; // fine channel

        float edgeFreq = low_freq + chan * coarse_chan_bw +
                         fine_chan * fine_chan_bw - fine_chan_bw * 0.5f;
        float phase = phase_prefactor / ((edgeFreq + fine_chan_bw) * edgeFreq *
                                         edgeFreq); // precalculate
        cufftComplex weight;
        __sincosf(phase,
                  &weight.y,
                  &weight.x); // TO DO: test if it is not approximate
        return weight;
    }
};

} // namespace kernels

void get_dm_responses(CoherentDedisperserConfig& config,
                      float dm_prefactor,
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
                                          config.low_freq,
                                          config.coarse_chan_bw,
                                          config.fine_chan_bw,
                                          dm_prefactor));
}

} // namespace skyweaver
