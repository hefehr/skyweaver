#include "cuComplex.h"
#include "psrdada_cpp/cuda_utils.hpp"
#include "skyweaver/CoherentBeamformer.cuh"

#include <cassert>

namespace skyweaver
{
namespace kernels
{

__forceinline__ __device__ void dp4a(int& c, const int& a, const int& b)
{
#if __CUDA_ARCH__ >= 610
    asm("dp4a.s32.s32 %0, %1, %2, %3;" : "+r"(c) : "r"(a), "r"(b), "r"(c));
#else
    char4& a4 = *((char4*)&a);
    char4& b4 = *((char4*)&b);
    c += a4.x * b4.x;
    c += a4.y * b4.y;
    c += a4.z * b4.z;
    c += a4.w * b4.w;
#endif
}

__forceinline__ __device__ int2 int2_transpose(int2 const& input)
{
    char2x4 a;
    char4x2 b;
    a     = (*(char2x4*)&input);
    b.x.x = a.x.x;
    b.x.y = a.y.x;
    b.x.z = a.z.x;
    b.x.w = a.w.x;
    b.y.x = a.x.y;
    b.y.y = a.y.y;
    b.y.z = a.z.y;
    b.y.w = a.w.y;
    return (*(int2*)&b);
}

__host__ __device__ static __inline__ float cuCmagf(cuFloatComplex x)
{
    return x.x * x.x + x.y * x.y;
}

/**
 * Stokes modes can be considered here:
 * I = P0^2 + P1^2
 * Q = P0^2 - P1^2
 * U = 2 * Re(P0 * conj(P1))
 * V = 2 * Im(P0 * conj(P1))
 */
__host__ __device__ static __inline__ float calculate_stokes(cuFloatComplex p0,
                                                             cuFloatComplex p1)
{
#if SKYWEAVER_STOKES_MODE == SKYWEAVER_STOKES_I
    return cuCmagf(p0) + cuCmagf(p1);
#elif SKYWEAVER_STOKES_MODE == SKYWEAVER_STOKES_Q
    return cuCmagf(p0) - cuCmagf(p1);
#elif SKYWEAVER_STOKES_MODE == SKYWEAVER_STOKES_U
    return 2 * cuCrealf(cuCmulf(p0, cuConjf(p1)));
#elif SKYWEAVER_STOKES_MODE == 3 SKYWEAVER_STOKES_V
    return 2 * cuCimagf(cuCmulf(p0, cuConjf(p1)));
#else
    static_assert(false,
                  "Invalid Stokes mode defined. Must be one "
                  "of 0 (I), 1 (Q), 2 (U) or 3 (V)");
#endif
}

__global__ void bf_aptf_general_k(int2 const* __restrict__ ftpa_voltages,
                                  int2 const* __restrict__ fbpa_weights,
                                  int8_t* __restrict__ tbtf_powers,
                                  float const* __restrict__ output_scale,
                                  float const* __restrict__ output_offset,
                                  int nsamples)
{
    /**
     * Perform compile time checks on requested beamforming parameters.
     */
    static_assert(SKYWEAVER_NBEAMS % SKYWEAVER_CB_WARP_SIZE == 0,
                  "Kernel can only process a multiple of 32 beams.");
    // This can no longer be a static assert as the NSAMPLES is no longer fixed
    // static_assert(NSAMPLES%SKYWEAVER_CB_NSAMPLES_PER_BLOCK==0,
    //    "Kernel can only process a multiple of (NWARPS_PER_BLOCK *
    //    SKYWEAVER_IB_TSCRUNCH) samples.");
    static_assert(SKYWEAVER_CB_NTHREADS % SKYWEAVER_CB_WARP_SIZE == 0,
                  "Number of threads must be an integer multiple of "
                  "SKYWEAVER_CB_WARP_SIZE.");
    static_assert(SKYWEAVER_NANTENNAS % 4 == 0,
                  "Number of antennas must be a multiple of 4.");
    static_assert(SKYWEAVER_NPOL == 2,
                  "This kernel only works for dual polarisation data.");

    /**
     * Allocated shared memory to store beamforming weights and temporary space
     * for antenna data.
     */
    __shared__ int2
        shared_apb_weights[SKYWEAVER_NANTENNAS / 4][SKYWEAVER_CB_WARP_SIZE];
    __shared__ int2
        shared_antennas[SKYWEAVER_CB_NTHREADS / SKYWEAVER_CB_WARP_SIZE]
                       [SKYWEAVER_NANTENNAS / 4];
    int const warp_idx = threadIdx.x / 0x20;
    int const lane_idx = threadIdx.x & 0x1f;
    /**
     * Each warp processes 32 beams (i.e. one beam per lane).
     */
    int const start_beam_idx = blockIdx.z * SKYWEAVER_CB_WARP_SIZE;

    /**
     * Complex multiply accumulators
     */
    int xx, yy, xy, yx;
    float power = 0.0f;

#if SKYWEAVER_IB_SUBTRACTION
    // Currently IB subtraction is implemented in this
    // kernel as a baseline. In future a version of this
    // kernel may be passed the IB data as an arg.
    // char4 ib_weight_c = {1, 1, 1, 1};
    // int ib_weight = *(int*)&ib_weight_c;
    // Simplifies to:
    int ib_weight = 0x1010101;
    int ib_xx, ib_yy;
    float ib_power = 0.0f;
#endif // SKYWEAVER_IB_SUBTRACTION

    int2 antennas, weights;
    int antenna_group_idx;
    const int sample_offset =
        SKYWEAVER_IB_TSCRUNCH *
        (blockIdx.x * SKYWEAVER_CB_NWARPS_PER_BLOCK + warp_idx);

    for(int channel_idx = blockIdx.y * SKYWEAVER_CB_FSCRUNCH;
        channel_idx < (blockIdx.y + 1) * SKYWEAVER_CB_FSCRUNCH;
        ++channel_idx) {
        /**
         * Here we load all the beamforming weights neccessary for this block.
         * Implicit assumption here is that we do not need to change the weights
         * over the timescale of the data processed in one block. This is almost
         * certainly OK if the input data has already been rotated to telescope
         * boresight and we are only applying parallactic angle tracking
         * updates.
         *
         * The global load is coalesced 8-byte (vectorised int2).
         */
        int const fbpa_weights_offset =
            SKYWEAVER_NANTENNAS / 4 *
            (SKYWEAVER_NBEAMS * channel_idx + (start_beam_idx + warp_idx));
        for(antenna_group_idx = lane_idx;
            antenna_group_idx < SKYWEAVER_NANTENNAS / 4;
            antenna_group_idx += SKYWEAVER_CB_WARP_SIZE) {
            shared_apb_weights[antenna_group_idx][warp_idx] = int2_transpose(
                fbpa_weights[fbpa_weights_offset + antenna_group_idx]);
        }

        // wait for all weights to load.
        __syncthreads();

        /**
         * Below is the main loop of the kernel. Here the kernel reads all the
         * antennas for a given sample and computes 32 beams. Each thread
         * computes only 1 beam and access to all the antennas required for that
         * computation is achieved via a shared memory broadcasts.
         */
        for(int sample_idx = sample_offset;
            sample_idx < (sample_offset + SKYWEAVER_IB_TSCRUNCH);
            ++sample_idx) {
            int ftpa_voltages_partial_idx =
                SKYWEAVER_NANTENNAS / 4 * SKYWEAVER_NPOL *
                (nsamples * channel_idx + sample_idx);

            // The loop below will compute the voltages for one pol/chan/sample
            // To make stokes vectors we need to keep this voltage in memory
            // to allow for the combination of the two polarisations.
            cuFloatComplex pol_voltage[SKYWEAVER_NPOL];
#if SKYWEAVER_IB_SUBTRACTION
            cuFloatComplex ib_pol_voltage[SKYWEAVER_NPOL];
#endif // SKYWEAVER_IB_SUBTRACTION

            for(int pol_idx = 0; pol_idx < SKYWEAVER_NPOL; ++pol_idx) {
                // Set the complex accumulator to zero before adding the next
                // polarisation
                xx = 0;
                yy = 0;
                xy = 0;
                yx = 0;
#if SKYWEAVER_IB_SUBTRACTION
                ib_xx = 0;
                ib_yy = 0;
#endif // SKYWEAVER_IB_SUBTRACTION

                /**
                 * Load all antennas antennas required for this sample into
                 * shared memory.
                 */
                if(lane_idx < SKYWEAVER_NANTENNAS / 4) {
                    shared_antennas[warp_idx][lane_idx] = int2_transpose(
                        ftpa_voltages[ftpa_voltages_partial_idx + lane_idx +
                                      SKYWEAVER_NANTENNAS / 4 * pol_idx]);
                }
                // Wait for all lanes to complete the load.
                __threadfence_block();
                for(antenna_group_idx = 0;
                    antenna_group_idx < SKYWEAVER_NANTENNAS / 4;
                    ++antenna_group_idx) {
                    // broadcast load 4 antennas
                    antennas = shared_antennas[warp_idx][antenna_group_idx];
                    // load corresponding 4 weights
                    weights = shared_apb_weights[antenna_group_idx][lane_idx];
                    // dp4a multiply add
                    dp4a(xx, weights.x, antennas.x);
                    dp4a(yy, weights.y, antennas.y);
                    dp4a(xy, weights.x, antennas.y);
                    dp4a(yx, weights.y, antennas.x);
#if SKYWEAVER_IB_SUBTRACTION
                    // Square the antenna signals and sum
                    dp4a(ib_xx, ib_weight, antennas.x);
                    dp4a(ib_yy, ib_weight, antennas.y);
#endif // SKYWEAVER_IB_SUBTRACTION
                }
                pol_voltage[pol_idx].x = (float)xx - (float)yy; // real
                pol_voltage[pol_idx].y = (float)xy + (float)yx; // imag
#if SKYWEAVER_IB_SUBTRACTION
                ib_pol_voltage[pol_idx].x = ib_xx;
                ib_pol_voltage[pol_idx].y = ib_yy;
#endif // SKYWEAVER_IB_SUBTRACTION
            }
            // This is after two polarisations have been computed
            power += calculate_stokes(pol_voltage[0], pol_voltage[1]);
#if SKYWEAVER_IB_SUBTRACTION
                ib_power +=
                calculate_stokes(ib_pol_voltage[0], ib_pol_voltage[1]);
#endif // SKYWEAVER_IB_SUBTRACTION
        }
        __syncthreads();
    }
    int const output_sample_idx = sample_offset / SKYWEAVER_IB_TSCRUNCH;
    int const tf_size           = SKYWEAVER_NSAMPLES_PER_HEAP * gridDim.y;
    int const btf_size          = gridDim.z * SKYWEAVER_CB_WARP_SIZE * tf_size;
    int const output_idx =
        (output_sample_idx / SKYWEAVER_NSAMPLES_PER_HEAP * btf_size +
         (start_beam_idx + lane_idx) * tf_size +
         (output_sample_idx % SKYWEAVER_NSAMPLES_PER_HEAP) * gridDim.y +
         blockIdx.y);
    float scale = output_scale[blockIdx.y];
#if SKYWEAVER_IB_SUBTRACTION
    /*
    Because we inflate the weights to have a magnitude of 127 to make sure
    that they can still represent many phases, we also need to account for
    this scaling factor in the incoherent beam.
    */
    float power_fp32 = rintf((power - ib_power * 127.0 * 127.0) / scale);
#else
    float power_fp32 = rintf((power - output_offset[blockIdx.y]) / scale);
#endif // SKYWEAVER_IB_SUBTRACTION
    tbtf_powers[output_idx] = (int8_t)fmaxf(-127.0f, fminf(127.0f, power_fp32));
}

} // namespace kernels

CoherentBeamformer::CoherentBeamformer(PipelineConfig const& config)
    : _config(config), _size_per_sample(0)
{
    BOOST_LOG_TRIVIAL(debug) << "Constructing CoherentBeamformer instance";
    _size_per_sample = _config.npol() * _config.nantennas() * _config.nchans();
    _expected_weights_size =
        _config.nbeams() * _config.nantennas() * _config.nchans();
    BOOST_LOG_TRIVIAL(debug) << "Size per sample: " << _size_per_sample;
    BOOST_LOG_TRIVIAL(debug)
        << "Expected weights size: " << _expected_weights_size;
}

CoherentBeamformer::~CoherentBeamformer()
{
}

void CoherentBeamformer::beamform(VoltageVectorType const& input,
                                  WeightsVectorType const& weights,
                                  ScalingVectorType const& output_scale,
                                  ScalingVectorType const& output_offset,
                                  PowerVectorType& output,
                                  cudaStream_t stream)
{
    assert(output_scale.size() == SKYWEAVER_NCHANS / SKYWEAVER_CB_FSCRUNCH /* Unexpected number of channels in scaling vector */);
    assert(output_offset.size() == SKYWEAVER_NCHANS / SKYWEAVER_CB_FSCRUNCH /* Unexpected number of channels in offset vector */);
    // First work out nsamples and resize output if not done already
    BOOST_LOG_TRIVIAL(debug) << "Executing coherent beamforming";
    assert(input.size() % _size_per_sample == 0);
    std::size_t nsamples = input.size() / _size_per_sample;
    std::size_t output_size =
        (input.size() / _config.nantennas() / _config.npol() /
         _config.cb_tscrunch() / _config.cb_fscrunch() * _config.nbeams());
    assert(nsamples % SKYWEAVER_CB_NSAMPLES_PER_BLOCK == 0);
    std::size_t nsamples_out = nsamples / _config.cb_tscrunch();
    assert(nsamples_out % SKYWEAVER_NSAMPLES_PER_HEAP == 0);
    BOOST_LOG_TRIVIAL(debug) << "Resizing output buffer from " << output.size()
                             << " to " << output_size << " elements";
    output.resize(output_size);
    assert(weights.size() == _expected_weights_size);
    dim3 grid(nsamples /
                  (SKYWEAVER_CB_NWARPS_PER_BLOCK * _config.cb_tscrunch()),
              _config.nchans() / _config.cb_fscrunch(),
              _config.nbeams() / SKYWEAVER_CB_WARP_SIZE);
    char2 const* ftpa_voltages_ptr = thrust::raw_pointer_cast(input.data());
    char2 const* fbpa_weights_ptr  = thrust::raw_pointer_cast(weights.data());
    int8_t* tbtf_powers_ptr        = thrust::raw_pointer_cast(output.data());
    float const* power_scaling = thrust::raw_pointer_cast(output_scale.data());
    float const* power_offset  = thrust::raw_pointer_cast(output_offset.data());
    BOOST_LOG_TRIVIAL(debug) << "Executing beamforming kernel";
    kernels::bf_aptf_general_k<<<grid, SKYWEAVER_CB_NTHREADS, 0, stream>>>(
        (int2 const*)ftpa_voltages_ptr,
        (int2 const*)fbpa_weights_ptr,
        tbtf_powers_ptr,
        power_scaling,
        power_offset,
        static_cast<int>(nsamples));
    CUDA_ERROR_CHECK(cudaStreamSynchronize(stream));
    BOOST_LOG_TRIVIAL(debug) << "Beamforming kernel complete";
}

} // namespace skyweaver
