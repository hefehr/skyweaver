#ifndef SKYWEAVER_COHERENTBEAMFORMER_CUH
#define SKYWEAVER_COHERENTBEAMFORMER_CUH

#include "skyweaver/PipelineConfig.hpp"
#include "skyweaver/beamformer_utils.cuh"
#include "thrust/device_vector.h"
#include "cuda.h"


namespace skyweaver {
namespace kernels {

/**
 * @brief      The coherent beamforming kernel
 *
 * @param      ftpa_voltages   The ftpa voltages (8 int8 complex values packed into int2)
 * @param      fbpa_weights    The fbpa weights (8 int8 complex values packed into int2)
 * @param      ftb_powers      The ftb powers
 * @param[in]  output_scale    The output scalings per channel
 * @param[in]  output_offset   The output offsets per channel
 * @param[in]  beamset_mapping The mapping of beam to beamset for choosing scalings
 * @param[in]  nsamples        The number of samples in the ftpa_voltages
 */
 template <typename BfTraits>
__global__ void bf_ftpa_general_k(int2 const* __restrict__ ftpa_voltages,
                                  int2 const* __restrict__ fbpa_weights,
                                  typename BfTraits::QuantisedPowerType* __restrict__ btf_powers,
                                  float const* __restrict__ output_scale,
                                  float const* __restrict__ output_offset,
                                  int const* __restrict__ beamset_mapping,
                                  typename BfTraits::RawPowerType const* __restrict__ ib_powers,
                                  int nsamples);

} //namespace kernels

/**
 * @brief      Class for coherent beamformer.
 */
template <typename BfTraits>
class CoherentBeamformer
{
public:
    // FTPA order
    typedef thrust::device_vector<char2> VoltageVectorType;
    // TBTF order
    typedef thrust::device_vector<typename BfTraits::QuantisedPowerType> PowerVectorType;
    typedef thrust::device_vector<typename BfTraits::RawPowerType> RawPowerVectorType;
    // FBA order (assuming equal weight per polarisation)
    typedef thrust::device_vector<char2> WeightsVectorType;
    typedef thrust::device_vector<float> ScalingVectorType;
    typedef thrust::device_vector<int> MappingVectorType;

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
     * @param      input     Input array of 8-bit voltages in FTPA order
     * @param      weights   8-bit beamforming weights in FTA order
     * @param      scales    Power scalings to be applied when converting data back to 8-bit
     * @param      offsets   Power offsets to be applied when converting data back to 8-bit
     * @param      output    Output array of 8-bit powers in TBTF order
     * @param      nbeamsets The number of beamsets being processed
     * @param[in]  stream    The CUDA stream to use for processing
     */
    void beamform(VoltageVectorType const& input,
        WeightsVectorType const& weights,
        ScalingVectorType const& scales,
        ScalingVectorType const& offsets,
        MappingVectorType const& beamset_mapping,
        RawPowerVectorType const& ib_powers,
        PowerVectorType& output,
        int nbeamsets,
        cudaStream_t stream);

private:
    PipelineConfig const& _config;
    std::size_t _size_per_sample;
    std::size_t _expected_weights_size;
};

extern template class CoherentBeamformer<SingleStokesBeamformerTraits<StokesParameter::I>>;
extern template class CoherentBeamformer<SingleStokesBeamformerTraits<StokesParameter::Q>>;
extern template class CoherentBeamformer<SingleStokesBeamformerTraits<StokesParameter::U>>;
extern template class CoherentBeamformer<SingleStokesBeamformerTraits<StokesParameter::V>>;
extern template class CoherentBeamformer<FullStokesBeamformerTraits>;

} //namespace skyweaver

#endif //SKYWEAVER_COHERENTBEAMFORMER_CUH
