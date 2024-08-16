#ifndef SKYWEAVER_INCOHERENTBEAMFORMER_CUH
#define SKYWEAVER_INCOHERENTBEAMFORMER_CUH

#include "beamformer_utils.cuh"
#include "cuda.h"
#include "psrdada_cpp/common.hpp"
#include "skyweaver/DescribedVector.hpp"
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
 * @param btf_powers_raw Output powers in BTF order and f32 (outer B is the
 * beamset)
 * @param btf_powers Output powers in BTF order and int8 (outer B is the
 * beamset)
 * @param output_scale Scale factor per frequency channel per beamset for 8-bit
 * conversion
 * @param output_offset Offset factor per frequency channel per beamset for
 * 8-bit conversion
 * @param antenna_weights Weights per antenna per beamset
 * @param nsamples Number of timesamples to process
 * @param nbeamsets Number of beamsets to process
 */
template <typename BfTraits>
__global__ void icbf_ftpa_general_k(
    char2 const* __restrict__ ftpa_voltages,
    typename BfTraits::RawPowerType* __restrict__ tf_powers_raw,
    typename BfTraits::QuantisedPowerType* __restrict__ tf_powers,
    float const* __restrict__ output_scale,
    float const* __restrict__ output_offset,
    float const* __restrict__ antenna_weights,
    int nsamples,
    int nbeamsets);

} // namespace kernels

template <typename BfTraits>
class IncoherentBeamformer
{
  public:
    typedef FTPAVoltagesD<char2> VoltageVectorTypeD;
    // TF order
    typedef BTFPowersD<typename BfTraits::QuantisedPowerType> PowerVectorTypeD;
    // TF order
    typedef BTFPowersD<typename BfTraits::RawPowerType> RawPowerVectorTypeD;
    // TF order
    typedef thrust::device_vector<float> ScalingVectorTypeD;

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
     * @param output_raw Output powers in BTF order where B is the number of
     * beamsets (floating point)
     * @param output Output powers in BTF order where B is the number of
     * beamsets (fixed point)
     * @param output_scale Scale factor per frequency channel per beamset for
     * fixed-point conversion
     * @param output_offset Offset factor per frequency channel per beamset for
     * fixed-point conversion
     * @param antenna_weights Weights per antenna per beamset
     * @param nbeamsets Number of beamsets to be handled (determined by the
     * delay model)
     * @param stream The CUDA stream to execute in
     */
    void beamform(VoltageVectorTypeD const& input,
                  RawPowerVectorTypeD& output_raw,
                  PowerVectorTypeD& output,
                  ScalingVectorTypeD const& output_scale,
                  ScalingVectorTypeD const& output_offset,
                  ScalingVectorTypeD const& antenna_weights,
                  int nbeamsets,
                  cudaStream_t stream);

  private:
    PipelineConfig const& _config;
};

extern template class IncoherentBeamformer<
    SingleStokesBeamformerTraits<StokesParameter::I>>;
extern template class IncoherentBeamformer<
    SingleStokesBeamformerTraits<StokesParameter::Q>>;
extern template class IncoherentBeamformer<
    SingleStokesBeamformerTraits<StokesParameter::U>>;
extern template class IncoherentBeamformer<
    SingleStokesBeamformerTraits<StokesParameter::V>>;
extern template class IncoherentBeamformer<
    StokesTraits<StokesParameter::Q, StokesParameter::U>>;
extern template class IncoherentBeamformer<
    StokesTraits<StokesParameter::I, StokesParameter::V>>;
extern template class IncoherentBeamformer<FullStokesBeamformerTraits>;

} // namespace skyweaver

#endif // SKYWEAVER_INCOHERENTBEAMFORMER_CUH
