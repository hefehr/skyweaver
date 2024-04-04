#ifndef SKY_WEAVER_OFFLINEPIPELINE_CUH
#define SKY_WEAVER_OFFLINEPIPELINE_CUH

#include "sky_weaver/cpp/fbfuse/PipelineConfig.hpp"
#include "sky_weaver/cpp/fbfuse/DelayManager.cuh"
#include "sky_weaver/cpp/fbfuse/WeightsManager.cuh"
#include "sky_weaver/cpp/fbfuse/GainManager.cuh"
#include "sky_weaver/cpp/fbfuse/ChannelScalingManager.cuh"
#include "sky_weaver/cpp/fbfuse/SplitTranspose.cuh"
#include "sky_weaver/cpp/fbfuse/CoherentBeamformer.cuh"
#include "sky_weaver/cpp/fbfuse/IncoherentBeamformer.cuh"
#include "sky_weaver/cpp/fbfuse/VoltageScaling.cuh"

#include "psrdada_cpp/dada_write_client.hpp"
#include "psrdada_cpp/double_device_buffer.cuh"
#include "psrdada_cpp/raw_bytes.hpp"
#include "cuda.h"
#include <memory>

namespace sky_weaver {

/**
 * @brief      Offline beamforming pipeline
 */
template<typename CBHandler, typename IBHandler>
class OfflinePipeline
{
public:
    typedef thrust::device_vector<char2> VoltageVectorType;
    typedef thrust::device_vector<int8_t> PowerVectorType;
    typedef thrust::device_vector<float> ChannelScaleVectorType;
    typedef long double TimeType;

public:
    /**
     * @brief      Constructs the pipeline object.
     *
     * @param      config                  The pipeline configuration
     * @param      cb_handler               DADA write client for output coherent beam data buffer
     * @param      ib_handler               DADA write client for output incoherent beam data buffer
     * @param[in]  input_data_buffer_size  The input DADA buffer block size
     */
    OfflinePipeline(PipelineConfig const& config,
        CBHandler& cb_handler,
        IBHandler& ib_handler,
        std::size_t input_data_buffer_size);
    ~OfflinePipeline();
    OfflinePipeline(Pipeline const&) = delete;

    /**
     * @brief      Initialise the pipeline with a DADA header block
     *
     * @param      header  A RawBytes object wrapping the DADA header block
     */
    void init(RawBytes& header);

    /**
     * @brief      Process the data in a DADA data buffer
     *
     * @param      data  A RawBytes object wrapping the DADA data block
     */
    bool operator()(RawBytes& data);

private:
    void process(VoltageVectorType&, PowerVectorType&, PowerVectorType&);
    void set_header(RawBytes& header);

private:
    PipelineConfig const& _config;

    // Data info
    std::size_t _sample_clock_start;
    long double _sample_clock;
    long double _sync_time;
    long double _unix_timestamp;
    std::size_t _sample_clock_tick_per_block;
    std::size_t _call_count;

    // Double buffers
    DoubleDeviceBuffer<char2> _taftp_db; // Input from F-engine
    DoubleDeviceBuffer<int8_t> _tbtf_db; // Output of coherent beamformer
    DoubleDeviceBuffer<int8_t> _tf_db; // Output of incoherent beamformer

    // Handlers
    CBHandler& _cb_handler;
    IBHandler& _ib_handler;

    // Streams
    cudaStream_t _h2d_copy_stream;
    cudaStream_t _processing_stream;
    cudaStream_t _d2h_copy_stream;

    // Data size info
    std::size_t _nheap_groups_per_block;
    std::size_t _nsamples_per_dada_block;

    // Pipeline components
    std::unique_ptr<DelayManager> _delay_manager;
    std::unique_ptr<WeightsManager> _weights_manager;
    std::unique_ptr<GainManager> _gain_manager;
    std::unique_ptr<ChannelScalingManager> _stats_manager;
    std::unique_ptr<SplitTranspose> _split_transpose;
    std::unique_ptr<CoherentBeamformer> _coherent_beamformer;
    std::unique_ptr<IncoherentBeamformer> _incoherent_beamformer;

    // Buffers
    VoltageVectorType _split_transpose_output;
    ChannelScaleVectorType _channel_scalings;
};

} //namespace sky_weaver

#include "sky_weaver/cpp/fbfuse/detail/OfflinePipeline.cuh"

#endif //SKY_WEAVER_OFFLINEPIPELINE_CUH
