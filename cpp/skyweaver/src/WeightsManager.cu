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

#ifndef SKYWEAVER_VISIBILITIES
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

    //  Reads and writes are no longer coalesced, sorry :(

    const int weights_per_beam = nvisibilities;
    // Positive root from quadratic formula for Nvis = Nant * (Nant - 1) / 2
    const int nantennas = (int) (-1 + sqrt(1 + 8 * (double) nvisibilities)) / 2;
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

            for(int antenna1_idx = threadIdx.x; antenna1_idx < nantennas;
                antenna1_idx += blockDim.x) {
                for(int antenna2_idx = antenna1_idx + 1;
                    antenna2_idx < nantennas; antenna2_idx += 1) {
                    float3 delay_model1 =
                        delay_models[beam_idx * weights_per_beam + antenna1_idx]; // correct
                    float3 delay_model2 =
                        delay_models[beam_idx * weights_per_beam + antenna2_idx]; // correct

                    double delay_offset1 = (double)delay_model1.y;
                    double delay_rate1   = (double)delay_model1.z;
                    double delay_offset2 = (double)delay_model2.y;
                    double delay_rate2   = (double)delay_model2.z;

                    int visibility_offset  = beam_offset + antenna1_idx * nantennas + antenna2_idx;
                    for(int time_idx = threadIdx.y; time_idx < ntsteps;
                        time_idx += blockDim.y) {
                        // Calculates epoch offset
                        double t = (current_epoch - delay_epoch) + time_idx * tstep;
                        double phase1 = (t * delay_rate1 + delay_offset1) * frequency;
                        double phase2 = (t * delay_rate2 + delay_offset2) * frequency;
                        // This is possible as the magnitude of the weight is 1
                        // If we ever have to implement scalar weightings, this
                        // must change.
                        sincos(TWOPI * (phase1 - phase2), &weight.y, &weight.x);
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
    dim3 block(32, 32, 1);
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
