#ifndef SKYWEAVER_INCOHERENTDEDISPERSIONPIPELINE_CUH
#define SKYWEAVER_INCOHERENTDEDISPERSIONPIPELINE_CUH

#include "skyweaver/IncoherentDedisperser.cuh"
#include "skyweaver/DescribedVector.hpp"
#include "skyweaver/AggregationBuffer.cuh"
#include "skyweaver/PipelineConfig.hpp"
#include "skyweaver/ObservationHeader.hpp"
#include <vector>
#include <functional>
#include <memory>

namespace skyweaver
{

template <typename InputType, typename OutputType, typename Handler>
class IncoherentDedispersionPipeline
{
public: 
    typedef AggregationBuffer<InputType> AggBufferType;
    typedef std::vector<std::unique_ptr<AggBufferType>> AggBufferVector;
    typedef TFBPowersH<InputType> InputVectorType; //CoherentBeam Data
    typedef TDBPowersH<OutputType> OutputVectorType; //Dedispersered Data
    typedef IncoherentDedisperser DedisperserType;
    typedef std::vector<std::unique_ptr<DedisperserType>> DedisperserVector;
    
public:
    IncoherentDedispersionPipeline(PipelineConfig const& config, Handler& handler);
    ~IncoherentDedispersionPipeline();
    IncoherentDedispersionPipeline(IncoherentDedispersionPipeline const&) = delete;
    IncoherentDedispersionPipeline& operator=(IncoherentDedispersionPipeline const&) = delete;
    void init(ObservationHeader const& header);
    void operator()(InputVectorType const& data, std::size_t dm_idx);

private:
    void agg_buffer_callback(typename InputVectorType::VectorType const& buffer, std::size_t dm_idx);

private:
    PipelineConfig const& _config;
    Handler& _handler;
    AggBufferVector _agg_buffers;
    DedisperserVector _dedispersers;
    std::vector<OutputVectorType> _output_buffers;
};

} // namespace skyweaver

#include "skyweaver/detail/IncoherentDedispersionPipeline.cu"

#endif // SKYWEAVER_INCOHERENTDEDISPERSIONPIPELINE_CUH