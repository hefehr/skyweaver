#ifndef SKYWEAVER_WEIGHTSMANAGER_CUH
#define SKYWEAVER_WEIGHTSMANAGER_CUH

#include "skyweaver/DelayManager.cuh"
#include "skyweaver/PipelineConfig.hpp"

#include <thrust/device_vector.h>

namespace skyweaver
{
namespace kernels
{

__global__ void
generate_weights_k(float3 const* __restrict__ delay_models,
                   char2* __restrict__ weights,
                   double const* __restrict__ channel_frequencies,
                   int nantennas,
                   int nbeams,
                   int nchans,
                   double tstart,
                   double tstep,
                   int ntsteps);

} // namespace kernels

class WeightsManager
{
  public:
    typedef char2 WeightsType;
    typedef thrust::device_vector<WeightsType> WeightsVectorTypeD;
    typedef double FreqType;
    typedef thrust::device_vector<FreqType> FreqVectorTypeD;
    typedef double TimeType;
    typedef DelayManager::DelayVectorTypeD DelayVectorTypeD;

  public:
    /**
     * @brief      Create a new weights mananger object
     *
     * @param      config          The pipeline configuration
     */
    WeightsManager(PipelineConfig const& config, cudaStream_t stream);
    ~WeightsManager();
    WeightsManager(WeightsManager const&) = delete;

    /**
     * @brief      Calculate beamforming weights for a given epoch
     *
     * @param[in]  delays A vector containing delay models for all beams
     * (produced by an instance of DelayManager)
     *
     * @param[in]  epoch  The epoch at which to evaluate the given delay models
     *
     * @details     No check is performed here on whether the provided epoch is
     * in the bounds of the current delay polynomial.
     *
     * @note       This function is not thread-safe!!! Competing calls will
     * overwrite the memory of the _weights object.
     *
     * @return     A thrust device vector containing the generated weights
     */
    WeightsVectorTypeD const& weights(DelayVectorTypeD const& delays,
                                      TimeType current_epoch,
                                      TimeType delay_epoch);

  private:
    PipelineConfig const& _config;
    cudaStream_t _stream;
    WeightsVectorTypeD _weights;
    FreqVectorTypeD _channel_frequencies;
};

} // namespace skyweaver

#endif // SKYWEAVER_WEIGHTSMANAGER_CUH