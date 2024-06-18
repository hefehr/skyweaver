#include "psrdada_cpp/cuda_utils.hpp"
#include "skyweaver/Transposer.cuh"

#include <limits>

#define SKYWEAVER_ST_MAX_ANTENNAS 32

namespace skyweaver
{
namespace kernels
{

__global__ void transpose_k(char2 const* __restrict__ input,
                            char2* __restrict__ output,
                            int input_nantennas,
                            int output_nantennas,
                            int nchans,
                            int ntimestamps)
{
    // Requires output_nantennas >= input_nantennas
    __shared__ char2
        transpose_buffer[SKYWEAVER_ST_MAX_ANTENNAS][SKYWEAVER_NSAMPLES_PER_HEAP]
                        [SKYWEAVER_NPOL];

    // TAFTP (input dimensions)
    const int tp   = SKYWEAVER_NSAMPLES_PER_HEAP * SKYWEAVER_NPOL;
    const int ftp  = nchans * tp;
    const int aftp = input_nantennas * ftp;

    // FTPA
    const int pa  = SKYWEAVER_NPOL * output_nantennas;
    const int tpa = ntimestamps * SKYWEAVER_NSAMPLES_PER_HEAP * pa;

    int nantennas_sets =
        ceilf(((float)input_nantennas) / SKYWEAVER_ST_MAX_ANTENNAS);

    // Each timestamp here is a heap timestamp so 256 samples
    for(int timestamp_idx = blockIdx.x; timestamp_idx < ntimestamps;
        timestamp_idx += gridDim.x) {
        // Each channel is a standard frequency channel
        for(int chan_idx = blockIdx.y; chan_idx < nchans;
            chan_idx += gridDim.y) {
            // We load antennas in groups of 32 for efficiency, each set is up
            // to 32 antennas
            for(int antenna_set_idx = 0; antenna_set_idx < nantennas_sets;
                ++antenna_set_idx) {
                // Here calculate the number of antennas in the current set
                int remaining_antennas =
                    min(input_nantennas -
                            antenna_set_idx * SKYWEAVER_ST_MAX_ANTENNAS,
                        SKYWEAVER_ST_MAX_ANTENNAS);

                // Load antenna data into shared memory
                for(int antenna_idx = threadIdx.y;
                    antenna_idx < remaining_antennas;
                    antenna_idx += blockDim.y) {
                    int input_antenna_idx =
                        antenna_set_idx * SKYWEAVER_ST_MAX_ANTENNAS +
                        antenna_idx;

                    // Loop over the TP samples perforing a coalesced read from
                    // global memory and a coalesced write to shared memory
                    for(int samppol_idx = threadIdx.x;
                        samppol_idx <
                        (SKYWEAVER_NSAMPLES_PER_HEAP * SKYWEAVER_NPOL);
                        samppol_idx += blockDim.x) {
                        int pol_idx  = samppol_idx % SKYWEAVER_NPOL;
                        int samp_idx = samppol_idx / SKYWEAVER_NPOL;
                        int input_idx =
                            timestamp_idx * aftp + input_antenna_idx * ftp +
                            chan_idx * tp + samp_idx * SKYWEAVER_NPOL + pol_idx;
                        transpose_buffer[antenna_idx][samp_idx][pol_idx] =
                            input[input_idx];
                    }
                }
                __syncthreads();
                for(int pol_idx = 0; pol_idx < SKYWEAVER_NPOL; ++pol_idx) {
                    for(int samp_idx = threadIdx.y;
                        samp_idx < SKYWEAVER_NSAMPLES_PER_HEAP;
                        samp_idx += blockDim.y) {
                        int output_sample_idx =
                            samp_idx +
                            timestamp_idx * SKYWEAVER_NSAMPLES_PER_HEAP;
                        for(int antenna_idx = threadIdx.x;
                            antenna_idx < remaining_antennas;
                            antenna_idx += blockDim.x) {
                            int output_antenna_idx =
                                antenna_set_idx * SKYWEAVER_ST_MAX_ANTENNAS +
                                antenna_idx;
                            // FTPA
                            int output_idx =
                                chan_idx * tpa + output_sample_idx * pa +
                                pol_idx * output_nantennas + output_antenna_idx;
                            output[output_idx] =
                                transpose_buffer[antenna_idx][samp_idx]
                                                [pol_idx];
                        }
                    }
                }
                __syncthreads();
            }
        }
    }
}

} // namespace kernels

Transposer::Transposer(PipelineConfig const& config)
    : _config(config), _output_size_per_heap_group(0)
{
    BOOST_LOG_TRIVIAL(debug) << "Constructing Transposer instance";
    _output_size_per_heap_group =
        (_config.npol() * _config.nsamples_per_heap() * _config.nchans() *
         _config.nantennas());
    BOOST_LOG_TRIVIAL(debug)
        << "Output size per heap group: " << _output_size_per_heap_group;
}

Transposer::~Transposer()
{
}

void Transposer::transpose(VoltageType const& taftp_voltages,
                           VoltageType& ftpa_voltages,
                           std::size_t input_nantennas,
                           cudaStream_t stream)
{
    BOOST_LOG_TRIVIAL(debug) << "Performing split transpose";

    std::size_t heap_group_size =
        (_config.npol() * _config.nsamples_per_heap() * _config.nchans() *
         input_nantennas);

    // Check sizes
    BOOST_LOG_TRIVIAL(debug)
        << "Transpose input size: " << taftp_voltages.size();
    if(taftp_voltages.size() >=
       std::numeric_limits<int>::
           max()) { /*
                    This check is needed as we use ints for indexes in the GPU
                    kernel. The int indexes are intended for performance but it
                    has not been tested to see what effect using uint64_t or
                    ptrdiff_t would have on this. This is a common problem for
                    GPU kernels, see e.g.
                    https://github.com/NVIDIA/cub/issues/212 and
                    https://stackoverflow.com/questions/14105958/cuda-c-best-practices-unsigned-vs-signed-optimization
                    */
        throw std::runtime_error(
            "TAFTP array is too large to be indexed with <int> indexes.");
    }
    if(taftp_voltages.size() % heap_group_size != 0) {
        throw std::runtime_error(
            "Voltages are not a multiple of the heap size");
    }
    if(input_nantennas > _config.nantennas()) {
        throw std::runtime_error(
            "Input number of antennas must be <= to the maximum nantennas");
    }
    int nheap_groups = taftp_voltages.size() / heap_group_size;
    BOOST_LOG_TRIVIAL(debug) << "Number of heap groups: " << nheap_groups;
    // Resize output buffer
    BOOST_LOG_TRIVIAL(debug)
        << "Resizing output buffer from " << ftpa_voltages.size() << " to "
        << _output_size_per_heap_group * nheap_groups;
    // Resize the output and set the data to zero
    // _output_size_per_heap_group contains the number of antennas to pad to
    ftpa_voltages.resize(_output_size_per_heap_group * nheap_groups, {0, 0});
    dim3 grid(nheap_groups, _config.nchans(), 1);
    dim3 block(512, 1, 1);
    char2 const* input_ptr = thrust::raw_pointer_cast(taftp_voltages.data());
    char2* output_ptr      = thrust::raw_pointer_cast(ftpa_voltages.data());
    BOOST_LOG_TRIVIAL(debug)
        << "Transposing and padding TAFTP (" << nheap_groups << ", "
        << input_nantennas << ", " << _config.nchans() << ", "
        << _config.nsamples_per_heap() << ", " << _config.npol()
        << ") to FTPA (" << _config.nchans() << ", "
        << _config.nsamples_per_heap() * nheap_groups << ", " << _config.npol()
        << ", " << _config.nantennas() << ")";
    BOOST_LOG_TRIVIAL(debug) << "Launching split transpose kernel";
    size_t mem_tot;
    size_t mem_free;
    cudaMemGetInfo(&mem_free, &mem_tot);
    BOOST_LOG_TRIVIAL(debug)
        << "Free memory: " << mem_free << " of " << mem_tot << " bytes\n";
    BOOST_LOG_TRIVIAL(debug) << input_ptr << ", " << output_ptr << "\n";
    kernels::transpose_k<<<grid, block, 0, stream>>>(input_ptr,
                                                     output_ptr,
                                                     input_nantennas,
                                                     _config.nantennas(),
                                                     _config.nchans(),
                                                     nheap_groups);
    // Not sure if this should be here, will check later
    CUDA_ERROR_CHECK(cudaStreamSynchronize(stream));
    BOOST_LOG_TRIVIAL(debug) << "Split transpose complete";
}

} // namespace skyweaver
