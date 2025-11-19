#include "psrdada_cpp/cuda_utils.hpp"
#include "skyweaver/DelayManager.cuh"
#include "skyweaver/PipelineConfig.hpp"
#include "skyweaver/WeightsManager.cuh"
#include "skyweaver/types.cuh"

#include <thrust/device_vector.h>

#define TWOPI 6.283185307179586

namespace skyweaver
{
namespace kernels
{

#if SKYWEAVER_VOLTAGES
__global__ void
generate_weights_k(float3 const* __restrict__ delay_models,
                   char2* __restrict__ weights,
                   double const* __restrict__ channel_frequencies,
                   int nantennas,
                   int nbeams,
                   int nchans,
                   double current_epoch,
                   double delay_epoch,
                   double tstep,
                   int ntsteps)
{
    // for each loaded delay poly we can produce multiple epochs for one
    // antenna, one beam, all frequencies and both pols Different blocks should
    // handle different beams (as antennas are on the inner dimension of the
    // output product)

    // Basics of this kernel:
    //
    //  gridDim.x is used for beams (there is a loop if you want to limit the
    //  grid size) gridDim.y is used for channels (there is a loop if you want
    //  to limit the grid size) blockDim.x is used for antennas (there is a loop
    //  if you want to limit the grid size)
    //
    //  Time steps are handled in a the inner loop. As antennas are on the inner
    //  dimension of both the input and the output array, all reads and writes
    //  should be coalesced.
    const int weights_per_beam      = nantennas;
    const int weights_per_channel   = weights_per_beam * nbeams;
    const int weights_per_time_step = weights_per_channel * nchans;

    double2 weight;
    char2 compressed_weight;
    // This isn't really needed as there will never be more than 64 antennas
    // However this makes this fucntion more flexible with smaller blocks
    for(int chan_idx = blockIdx.y; chan_idx < nchans; chan_idx += gridDim.y) {
        double frequency = channel_frequencies[chan_idx];
        int chan_offset  = chan_idx * weights_per_channel; // correct

        for(int beam_idx = blockIdx.x; beam_idx < nbeams;
            beam_idx += gridDim.x) {
            int beam_offset =
                chan_offset + beam_idx * weights_per_beam; // correct

            for(int antenna_idx = threadIdx.x; antenna_idx < nantennas;
                antenna_idx += blockDim.x) {
                float3 delay_model =
                    delay_models[beam_idx * nantennas + antenna_idx]; // correct
                double delay_offset = (double)delay_model.y;
                double delay_rate   = (double)delay_model.z;
                int antenna_offset  = beam_offset + antenna_idx;
                for(int time_idx = threadIdx.y; time_idx < ntsteps;
                    time_idx += blockDim.y) {
                    // Calculates epoch offset
                    double t = (current_epoch - delay_epoch) + time_idx * tstep;
                    double phase = (t * delay_rate + delay_offset) * frequency;
                    // This is possible as the magnitude of the weight is 1
                    // If we ever have to implement scalar weightings, this
                    // must change.
                    sincos(TWOPI * phase, &weight.y, &weight.x);
                    compressed_weight.x = clamp<int8_t, int>(
                        __double2int_rn(weight.x * 127.0 * delay_model.x));
                    compressed_weight.y = clamp<int8_t, int>(__double2int_rn(
                        -1.0 * weight.y * 127.0 * delay_model.x));
                    int output_idx =
                        time_idx * weights_per_time_step + antenna_offset;
                    weights[output_idx] = compressed_weight;
                }
            }
        }
    }
}

#else
__global__ void
    generate_weights_k(float3 const* __restrict__ delay_models,
                       char2* __restrict__ weights,
                       double const* __restrict__ channel_frequencies,
                       int nvisibilities,
                       int nbeams,
                       int nchans,
                       double current_epoch,
                       double delay_epoch,
                       double tstep,
                       int ntsteps)
{
    // for each loaded delay poly we can produce multiple epochs for one
    // baseline, one beam, all frequencies and both pols Different blocks should
    // handle different beams (as antennas are on the inner dimension of the
    // output product)

    // Basics of this kernel:
    //
    //  gridDim.x is used for beams (there is a loop if you want to limit the
    //  grid size) gridDim.y is used for channels (there is a loop if you want
    //  to limit the grid size) blockDim.x is used for baselines (there is a loop
    //  if you want to limit the grid size)
    //
    //  Time steps are handled in a the inner loop.

    //  Reads and writes are coalesced

    __shared__ float3 shared_delays[64];

    const int nantennas = (int) sqrt(2 * nvisibilities);
    const int weights_per_beam = nvisibilities;
    const int weights_per_channel   = weights_per_beam * nbeams;
    const int weights_per_time_step = weights_per_channel * nchans;

    double2 weight;
    char2 compressed_weight;
    // This isn't really needed as there will never be more than 64 antennas
    // However this makes this fucntion more flexible with smaller blocks
    for(int chan_idx = blockIdx.y; chan_idx < nchans; chan_idx += gridDim.y) {
        double frequency = channel_frequencies[chan_idx];
        int chan_offset  = chan_idx * weights_per_channel;

        for(int beam_idx = blockIdx.x; beam_idx < nbeams;
            beam_idx += gridDim.x) {
            int beam_offset =
                chan_offset + beam_idx * weights_per_beam;

            // Load delay models into shared memory
            if (threadIdx.y == 0) {
                for (int didx = threadIdx.x; didx < nantennas; didx += blockDim.x){
                    shared_delays[didx] = delay_models[beam_idx * weights_per_beam + didx];
                }
            }
            __syncthreads();

            for (int vidx = threadIdx.x + threadIdx.y * blockDim.x;
                 vidx < nantennas * (nantennas - 1) / 2;
                 vidx += blockDim.x * blockDim.y) {

                const int visibility_offset = beam_offset + vidx;

                // Finding row index by solving quadratic equation
                const int a1idx = (int) (nantennas - 0.5f * (1 + sqrtf(1 + 4 * nantennas * nantennas
                                                                       - 4 * nantennas
                                                                       - 8 * vidx)));

                const int a2idx = vidx + a1idx * (1 - nantennas) + a1idx * (a1idx + 1) / 2 + 1;

                float3 delay_model1 = shared_delays[a1idx];
                double delay_offset1 = (double)delay_model1.y;
                double delay_rate1   = (double)delay_model1.z;

                float3 delay_model2 = shared_delays[a2idx];
                double delay_offset2 = (double)delay_model2.y;
                double delay_rate2   = (double)delay_model2.z;

                for(int time_idx = blockIdx.z; time_idx < ntsteps;
                    time_idx += gridDim.z) {
                    // Calculates epoch offset
                    double t = (current_epoch - delay_epoch) + time_idx * tstep;

                    // Phase difference between the two antennas
                    double phase = ((t * delay_rate1 + delay_offset1)
                                    - (t * delay_rate2 + delay_offset2)) * frequency;

                    // This is possible as the magnitude of the weight is 1
                    // If we ever have to implement scalar weightings, this
                    // must change.
                    sincos(TWOPI * phase, &weight.y, &weight.x);
                    compressed_weight.x = clamp<int8_t, int>(
                        __double2int_rn(weight.x * 127.0 * delay_model1.x * delay_model2.x));
                    compressed_weight.y = clamp<int8_t, int>(__double2int_rn(
                                                                 -1.0 * weight.y * 127.0 * delay_model1.x * delay_model2.x));
                    int output_idx =
                        time_idx * weights_per_time_step + visibility_offset;

                    weights[output_idx] = compressed_weight;
                }
            }
        }
    }
}
#endif

} // namespace kernels

WeightsManager::WeightsManager(PipelineConfig const& config,
                               cudaStream_t stream)
    : _config(config), _stream(stream)
{
    BOOST_LOG_TRIVIAL(debug)
        << "Constructing WeightsManager instance to hold weights for "
        << _config.nbeams() << " beams and " << _config.nantennas()
        << " antennas";
    _weights.resize(_config.nbeams() * _config.nantennas() * _config.nchans());
    char2 zero;
    zero.x = 0;
    zero.y = 0;
    thrust::fill(_weights.begin(),
                 _weights.end(),
                 zero);

    // This should be an implicit copy to the device
    BOOST_LOG_TRIVIAL(debug) << "Copying channel frequencies to the GPU";
    _channel_frequencies = _config.channel_frequencies();
}

WeightsManager::~WeightsManager()
{
}

WeightsManager::WeightsVectorTypeD const&
WeightsManager::weights(DelayVectorTypeD const& delays,
                        TimeType current_epoch,
                        TimeType delay_epoch)
{
    // First we retrieve new delays if there are any.
    BOOST_LOG_TRIVIAL(debug)
        << "Requesting weights: current epoch = " << current_epoch
        << ", delay model epoch = " << delay_epoch
        << " (difference = " << (current_epoch - delay_epoch) << ")";
    WeightsType* weights_ptr = thrust::raw_pointer_cast(_weights.data());
    FreqType const* frequencies_ptr =
        thrust::raw_pointer_cast(_channel_frequencies.data());
    dim3 grid(_config.nbeams(), _channel_frequencies.size(), 1);

#if SKYWEAVER_VOLTAGES
    dim3 block(32, 32, 1);
#else
    dim3 block(64, 4, 1);
#endif
    BOOST_LOG_TRIVIAL(debug) << "Launching weights generation kernel";
    kernels::generate_weights_k<<<grid, block, 0, _stream>>>(
        thrust::raw_pointer_cast(delays.data()),
        weights_ptr,
        frequencies_ptr,
        _config.nantennas(),
        _config.nbeams(),
        _channel_frequencies.size(),
        current_epoch,
        delay_epoch,
        0.0,
        1);
    CUDA_ERROR_CHECK(cudaStreamSynchronize(_stream));
    BOOST_LOG_TRIVIAL(debug) << "Weights successfully generated";
    return _weights;
}

} // namespace skyweaver
