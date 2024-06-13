#ifndef SKYWEAVER_INCOHERENTBEAMFORMER_CUH
#define SKYWEAVER_INCOHERENTBEAMFORMER_CUH

#include "cuda.h"
#include "psrdada_cpp/common.hpp"
#include "skyweaver/PipelineConfig.hpp"
#include "thrust/device_vector.h"

namespace skyweaver
{
namespace kernels
{

/**
 * @brief 
 * 
 * @param ftpa_voltages Input voltages in FTPA order
 * @param btf_powers_raw Output powers in BTF order and f32 (outer B is the beamset)
 * @param btf_powers Output powers in BTF order and int8 (outer B is the beamset)
 * @param output_scale Scale factor per frequency channel per beamset for 8-bit conversion
 * @param output_offset Offset factor per frequency channel per beamset for 8-bit conversion
 * @param antenna_weights Weights per antenna per beamset
 * @param nsamples Number of timesamples to process
 * @param nbeamsets Number of beamsets to process
 */
__global__ void icbf_ftpa_general_k(char2 const* __restrict__ ftpa_voltages,
                                    float* __restrict__ tf_powers_raw,
                                    int8_t* __restrict__ tf_powers,
                                    float const* __restrict__ output_scale,
                                    float const* __restrict__ output_offset,
                                    float const* __restrict__ antenna_weights,
                                    int nsamples,
                                    int nbeamsets);                              

} // namespace kernels

/**
 * @brief      Class for incoherent beamforming.
 */
class IncoherentBeamformer
{
  public:
    // TAFTP order
    typedef thrust::device_vector<char2> VoltageVectorType;
    // TF order
    typedef thrust::device_vector<int8_t> PowerVectorType;
    // TF order
    typedef thrust::device_vector<float> RawPowerVectorType;
    // TF order
    typedef thrust::device_vector<float> ScalingVectorType;

  public:
    /**
     * @brief      Constructs an instance of the IncoherentBeamformer class
     *
     * @param      config  The pipeline config
     */
    IncoherentBeamformer(PipelineConfig const& config);
    ~IncoherentBeamformer();
    IncoherentBeamformer(IncoherentBeamformer const&) = delete;

    /**
     * @brief Incoherently beamformer antenna voltages
     * 
     * @param input Input voltages in FTPA order
     * @param output_raw Output powers in BTF order where B is the number of beamsets (floating point)
     * @param output Output powers in BTF order where B is the number of beamsets (fixed point)
     * @param output_scale Scale factor per frequency channel per beamset for fixed-point conversion
     * @param output_offset Offset factor per frequency channel per beamset for fixed-point conversion
     * @param antenna_weights Weights per antenna per beamset
     * @param nbeamsets Number of beamsets to be handled (determined by the delay model)
     * @param stream The CUDA stream to execute in
     */
    void beamform(VoltageVectorType const& input,
                                    RawPowerVectorType& output_raw,
                                    PowerVectorType& output,
                                    ScalingVectorType const& output_scale,
                                    ScalingVectorType const& output_offset,
                                    ScalingVectorType const& antenna_weights,
                                    int nbeamsets,
                                    cudaStream_t stream);

  private:
    PipelineConfig const& _config;
};

} // namespace skyweaver

#endif // SKYWEAVER_INCOHERENTBEAMFORMER_CUH
