#include "psrdada_cpp/cuda_utils.hpp"
#include "skyweaver/IncoherentBeamformer.cuh"
#include "skyweaver/beamformer_utils.cuh"

#include <cassert>

namespace skyweaver
{
namespace kernels
{

__global__ void icbf_ftpa_general_k(char2 const* __restrict__ ftpa_voltages,
                                    float* __restrict__ tf_powers_raw,
                                    int8_t* __restrict__ tf_powers,
                                    float const* __restrict__ output_scale,
                                    float const* __restrict__ output_offset,
                                    int ntimestamps)
{
    // FTPA beamformer
    // Each thread loads the two pols of 1 antenna
    // Detects stokes
    // Sums over tscrunch and fscrunch
    // Total number of threads = N where N is a multiple of Nantennas
    // grid size = nchans/fscrunch, nsamples/tscrunch
    static_assert(SKYWEAVER_NPOL == 2,
                  "icbf_ftpa_general_k only works with dual pol data");

    const int a   = SKYWEAVER_NANTENNAS;
    const int pa  = SKYWEAVER_NPOL * a;
    const int tpa = ntimestamps * pa;

    volatile __shared__ float acc_buffer[SKYWEAVER_NANTENNAS];

    const int output_t_idx = blockIdx.x;
    const int output_f_idx = blockIdx.y;
    const int input_t_idx  = output_t_idx * SKYWEAVER_IB_TSCRUNCH;
    const int input_f_idx  = output_f_idx * SKYWEAVER_IB_FSCRUNCH;
    const int a_idx        = threadIdx.x;
    float power            = 0.0f;
    for(int f_idx = input_f_idx; f_idx < input_f_idx + SKYWEAVER_CB_FSCRUNCH;
        ++f_idx) {
        for(int t_idx = input_t_idx;
            t_idx < input_t_idx + SKYWEAVER_IB_TSCRUNCH;
            ++t_idx) {
            int input_p0_idx = f_idx * tpa + t_idx * pa + a_idx;
            int input_p1_idx = f_idx * tpa + t_idx * pa + a + a_idx;
            char2 p0_v       = ftpa_voltages[input_p0_idx];
            char2 p1_v       = ftpa_voltages[input_p1_idx];
            cuFloatComplex p0 =
                make_cuFloatComplex((float)p0_v.x, (float)p0_v.y);
            cuFloatComplex p1 =
                make_cuFloatComplex((float)p1_v.x, (float)p1_v.y);
            power += calculate_stokes(p0, p1);
        }
    }
    acc_buffer[threadIdx.x] = power;
    for(unsigned int ii = blockDim.x / 2; ii > 0; ii >>= 1) {
        __syncthreads();
        if(threadIdx.x < ii) {
            // Changed from += due to warning #3012-D
            acc_buffer[threadIdx.x] =
                acc_buffer[threadIdx.x] + acc_buffer[threadIdx.x + ii];
        }
    }
    __syncthreads();
    if(threadIdx.x == 0) {
        float power_f32           = acc_buffer[threadIdx.x];
        int output_idx            = output_t_idx * gridDim.y + output_f_idx;
        tf_powers_raw[output_idx] = power_f32;
        float scale               = output_scale[output_f_idx];
        float offset              = output_offset[output_f_idx];
        tf_powers[output_idx] =
            (int8_t)fmaxf(-127.0f,
                          fminf(127.0f, rintf((power_f32 - offset) / scale)));
    }
}

__global__ void icbf_taftp_general_k(char4 const* __restrict__ taftp_voltages,
                                     float* __restrict__ tf_powers_raw,
                                     int8_t* __restrict__ tf_powers,
                                     float const* __restrict__ output_scale,
                                     float const* __restrict__ output_offset,
                                     int ntimestamps)
{
    // TAFTP
    const int tp         = SKYWEAVER_NSAMPLES_PER_HEAP;
    const int ftp        = SKYWEAVER_NCHANS * tp;
    const int aftp       = SKYWEAVER_NANTENNAS * ftp;
    const int nchans_out = SKYWEAVER_NCHANS / SKYWEAVER_IB_FSCRUNCH;
    const int nsamps_out = SKYWEAVER_NSAMPLES_PER_HEAP / SKYWEAVER_IB_TSCRUNCH;
    volatile __shared__ float acc_buffer[SKYWEAVER_NSAMPLES_PER_HEAP];
    volatile __shared__ float output_buffer_raw[nsamps_out * nchans_out];
    volatile __shared__ int8_t output_buffer[nsamps_out * nchans_out];

    for(int timestamp_idx = blockIdx.x; timestamp_idx < ntimestamps;
        timestamp_idx += gridDim.x) {
        for(int start_channel_idx = 0; start_channel_idx < SKYWEAVER_NCHANS;
            start_channel_idx += SKYWEAVER_IB_FSCRUNCH) {
            float power = 0.0f;
            for(int sub_channel_idx = start_channel_idx;
                sub_channel_idx < start_channel_idx + SKYWEAVER_IB_FSCRUNCH;
                ++sub_channel_idx) {
                for(int antenna_idx = 0; antenna_idx < SKYWEAVER_NANTENNAS;
                    ++antenna_idx) {
                    int input_index = timestamp_idx * aftp + antenna_idx * ftp +
                                      sub_channel_idx * tp + threadIdx.x;

                    // Each ant here is both polarisations for one antenna
                    char4 ant = taftp_voltages[input_index];
                    cuFloatComplex p0 =
                        make_cuFloatComplex((float)ant.x, (float)ant.y);
                    cuFloatComplex p1 =
                        make_cuFloatComplex((float)ant.z, (float)ant.w);
                    power += calculate_stokes(p0, p1);
                }
            }
            acc_buffer[threadIdx.x] = power;
            __syncthreads();
            for(int ii = 1; ii < SKYWEAVER_IB_TSCRUNCH; ++ii) {
                int idx = threadIdx.x + ii;
                if(idx < SKYWEAVER_NSAMPLES_PER_HEAP) {
                    power += acc_buffer[idx];
                }
            }
            if(threadIdx.x % SKYWEAVER_IB_TSCRUNCH == 0) {
                int output_buffer_idx =
                    threadIdx.x / SKYWEAVER_IB_TSCRUNCH * nchans_out +
                    start_channel_idx / SKYWEAVER_IB_FSCRUNCH;
                float scale =
                    output_scale[start_channel_idx / SKYWEAVER_IB_FSCRUNCH];
                float offset =
                    output_offset[start_channel_idx / SKYWEAVER_IB_FSCRUNCH];
                output_buffer_raw[output_buffer_idx] = power;
                output_buffer[output_buffer_idx]     = (int8_t)fmaxf(
                    -127.0f,
                    fminf(127.0f, rintf((power - offset) / scale)));
            }
            __syncthreads();
        }
        int output_offset = timestamp_idx * nsamps_out * nchans_out;
        for(int idx = threadIdx.x;
            idx < nsamps_out * SKYWEAVER_NCHANS / SKYWEAVER_IB_FSCRUNCH;
            idx += blockDim.x) {
            tf_powers_raw[output_offset + idx] = output_buffer_raw[idx];
            tf_powers[output_offset + idx]     = output_buffer[idx];
        }
        __syncthreads();
    }
}

} // namespace kernels

IncoherentBeamformer::IncoherentBeamformer(PipelineConfig const& config)
    : _config(config)
{
    BOOST_LOG_TRIVIAL(debug) << "Constructing IncoherentBeamformer instance";
}

IncoherentBeamformer::~IncoherentBeamformer()
{
}

void IncoherentBeamformer::beamform(VoltageVectorType const& input,
                                    RawPowerVectorType& output_raw,
                                    PowerVectorType& output,
                                    ScalingVectorType const& output_scale,
                                    ScalingVectorType const& output_offset,
                                    cudaStream_t stream)
{
    // First work out nsamples and resize output if not done already
    BOOST_LOG_TRIVIAL(debug) << "Executing incoherent beamforming";
    const std::size_t fpa_size =
        _config.npol() * _config.nantennas() * _config.nchans();
    assert(input.size() % fpa_size == 0 /* Non integer number of time stamps*/);
    std::size_t ntimestamps = input.size() / fpa_size;
    std::size_t output_size =
        (input.size() / _config.nantennas() / _config.npol() /
         _config.ib_tscrunch() / _config.ib_fscrunch());
    BOOST_LOG_TRIVIAL(debug) << "Resizing output buffer from " << output.size()
                             << " to " << output_size << " elements";
    output.resize(output_size);
    output_raw.resize(output_size);
    assert(output_scale.size() == SKYWEAVER_NCHANS / SKYWEAVER_IB_FSCRUNCH /* Unexpected number of channels in scaling vector */);
    assert(output_offset.size() == SKYWEAVER_NCHANS / SKYWEAVER_IB_FSCRUNCH /* Unexpected number of channels in offset vector */);
    dim3 block(SKYWEAVER_NANTENNAS);
    dim3 grid(ntimestamps / SKYWEAVER_IB_TSCRUNCH,
              SKYWEAVER_NCHANS / SKYWEAVER_IB_FSCRUNCH);
    char2 const* ftpa_voltages_ptr = thrust::raw_pointer_cast(input.data());
    float const* output_scale_ptr =
        thrust::raw_pointer_cast(output_scale.data());
    float const* output_offset_ptr =
        thrust::raw_pointer_cast(output_offset.data());
    PowerVectorType::value_type* tf_powers_ptr =
        thrust::raw_pointer_cast(output.data());
    RawPowerVectorType::value_type* tf_powers_raw_ptr =
        thrust::raw_pointer_cast(output_raw.data());
    BOOST_LOG_TRIVIAL(debug) << "Executing incoherent beamforming kernel";
    kernels::icbf_ftpa_general_k<<<grid, block, 0, stream>>>(
        ftpa_voltages_ptr,
        tf_powers_raw_ptr,
        tf_powers_ptr,
        output_scale_ptr,
        output_offset_ptr,
        static_cast<int>(ntimestamps));
    CUDA_ERROR_CHECK(cudaStreamSynchronize(stream));
    BOOST_LOG_TRIVIAL(debug) << "Incoherent beamforming kernel complete";
}

} // namespace skyweaver
