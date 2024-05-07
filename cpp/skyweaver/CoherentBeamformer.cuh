#ifndef SKYWEAVER_COHERENTBEAMFORMER_CUH
#define SKYWEAVER_COHERENTBEAMFORMER_CUH

#include "skyweaver/PipelineConfig.hpp"
#include "thrust/device_vector.h"
#include "cuda.h"


namespace skyweaver {
namespace kernels {

/**
 * @brief      Data structure for holding reshaped int8 complex data
 *             for use in the DP4A transform. 
 * 
 * @details    Typically this stores the data from one polarisation for 
 *             4 antennas in r0,r1,r2,r3,i0,i1,i2,i3 order.
 */
struct char4x2
{
    char4 x;
    char4 y;
};


/**
 * @brief      Data structure for holding antenna voltages for transpose.
 * 
 * @details    Typically this stores the data from one polarisation for 
 *             4 antennas in r0,i0,r1,i1,r2,i2,r3,i3 order.
 */
struct char2x4
{
    char2 x;
    char2 y;
    char2 z;
    char2 w;
};

/**
 * @brief      Wrapper for the DP4A int8 fused multiply add instruction
 *
 * @param      c     The output value
 * @param[in]  a     An integer composed of 4 chars
 * @param[in]  b     An integer composed of 4 chars
 *
 * @detail     If we treat a and b like to char4 instances, then the dp4a
 *             instruction performs the following:
 *
 *             c = (a.x * b.x) + (a.y * b.y) + (a.z * b.z) + (a.w * b.w)
 *
 * @note       The assembly instruction that underpins this operation (dp4a.s32.s32).
 *             
 */
__forceinline__ __device__
void dp4a(int &c, const int &a, const int &b);

/**
 * @brief      Transpose an int2 from a char2x4 to a char4x2.
 *
 * @param      input  The value to transpose
 *
 * @note       This is used to go from (for 4 sequential antennas):
 *
 *             [[real, imag],
 *              [real, imag],
 *              [real, imag],
 *              [real, imag]]
 *
 *             to
 *
 *             [[real, real, real, real],
 *              [imag, imag, imag, imag]]
 */
__forceinline__ __device__
int2 int2_transpose(int2 const &input);

/**
 * @brief      The coherent beamforming kernel
 *
 * @param      ftpa_voltages  The ftpa voltages (8 int8 complex values packed into int2)
 * @param      fbpa_weights   The fbpa weights (8 int8 complex values packed into int2)
 * @param      ftb_powers     The ftb powers
 * @param[in]  output_scale   The output scaling
 * @param[in]  output_offset  The output offset
 * @param[in]  nsamples       The number of samples in the ftpa_voltages
 */
__global__
void bf_ftpa_general_k(
    int2 const* __restrict__ ftpa_voltages,
    int2 const* __restrict__ fbpa_weights,
    int8_t* __restrict__ ftb_powers,
    float output_scale,
    float output_offset,
    int nsamples);

} //namespace kernels

/**
 * @brief      Class for coherent beamformer.
 */
class CoherentBeamformer
{
public:
    // FTPA order
    typedef thrust::device_vector<char2> VoltageVectorType;
    // TBTF order
    typedef thrust::device_vector<int8_t> PowerVectorType;
    // FBA order (assuming equal weight per polarisation)
    typedef thrust::device_vector<char2> WeightsVectorType;
    typedef thrust::device_vector<float> ScalingVectorType;

public:
    /**
     * @brief      Constructs a CoherentBeamformer object.
     *
     * @param      config  The pipeline configuration
     */
    CoherentBeamformer(PipelineConfig const& config);
    ~CoherentBeamformer();
    CoherentBeamformer(CoherentBeamformer const&) = delete;

    /**
     * @brief      Form coherent beams
     *
     * @param      input    Input array of 8-bit voltages in FTPA order
     * @param      weights  8-bit beamforming weights in FTA order
     * @param      scales   Power scalings to be applied when converting data back to 8-bit
     * @param      offsets  Power offsets to be applied when converting data back to 8-bit
     * @param      output   Output array of 8-bit powers in TBTF order
     * @param[in]  stream   The CUDA stream to use for processing
     */
    void beamform(VoltageVectorType const& input,
        WeightsVectorType const& weights,
        ScalingVectorType const& scales,
        ScalingVectorType const& offsets,
        PowerVectorType& output,
        cudaStream_t stream);

private:
    PipelineConfig const& _config;
    std::size_t _size_per_sample;
    std::size_t _expected_weights_size;
};

} //namespace skyweaver

#endif //SKYWEAVER_COHERENTBEAMFORMER_CUH
