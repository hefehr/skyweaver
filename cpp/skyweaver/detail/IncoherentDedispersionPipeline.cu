#include "skyweaver/IncoherentDedispersionPipeline.cuh"

namespace skyweaver 
{

template <typename InputType, typename OutputType, typename Handler>
IncoherentDedispersionPipeline<
    InputType, OutputType, Handler
>::IncoherentDedispersionPipeline(PipelineConfig const& config, Handler& handler)
: _config(config), _handler(handler)
{   
    BOOST_LOG_TRIVIAL(info) << "Preparing incoherent dedispersion pipeline";
    auto const& dms = _config.coherent_dms();
    _dedispersers.resize(dms.size());
    _agg_buffers.resize(dms.size());
    _output_buffers.resize(dms.size());
    for (std::size_t dm_idx=0; dm_idx < dms.size(); ++dm_idx)
    {
        _dedispersers[dm_idx].reset(new DedisperserType(_config, dms[dm_idx]));
        int max_delay = _dedispersers[dm_idx]->max_delay();
        BOOST_LOG_TRIVIAL(debug) << "Created dedisperser for DM = " << dms[dm_idx] << " (max_delay = " << max_delay << ")";
        std::size_t dispatch_size = std::max(max_delay * 10, 8192);
        std::size_t batch_size = _config.nbeams() * _config.nchans();
        _agg_buffers[dm_idx].reset(new AggBufferType(
            std::bind(&IncoherentDedispersionPipeline::agg_buffer_callback, 
                        this, std::placeholders::_1, dm_idx), 
            dispatch_size, max_delay, batch_size));
        BOOST_LOG_TRIVIAL(debug) << "Created aggregation buffer for DM = " << dms[dm_idx] 
                                 << " (dispatch_size = " << dispatch_size << ", " 
                                 << "overlap_size = " << max_delay << ", "
                                 << "batch_size = " << batch_size << ")"; 
            //TODO: give these sensible numbers
            //TODO: work out how to get the max delay into the handler
    }
}

template <typename InputType, typename OutputType, typename Handler>
IncoherentDedispersionPipeline<InputType, OutputType, Handler>::~IncoherentDedispersionPipeline(){

}

template <typename InputType, typename OutputType, typename Handler>
void IncoherentDedispersionPipeline<InputType, OutputType, Handler>::agg_buffer_callback(typename InputVectorType::VectorType const& buffer, std::size_t dm_idx)
{
    BOOST_LOG_TRIVIAL(debug) << "Agg buffer callback called for dm_idx = " << dm_idx;
    _dedispersers[dm_idx]->dedisperse(buffer, _output_buffers[dm_idx]);
    BOOST_LOG_TRIVIAL(debug) << "Dedispersion complete, calling handler";
    BOOST_LOG_TRIVIAL(debug) << _output_buffers[dm_idx].vector().size();
    _handler(_output_buffers[dm_idx], dm_idx);
}

template <typename InputType, typename OutputType, typename Handler>
void IncoherentDedispersionPipeline<InputType, OutputType, Handler>::init(ObservationHeader const& header)
{
    std::size_t nchans = _config.channel_frequencies().size();
    long double chbw = _config.bandwidth() / _config.nchans();
    long double tsamp = _config.cb_tscrunch() / chbw;
    std::vector<long double> dm_delays(_config.coherent_dms().size());
    for (std::size_t dm_idx=0; dm_idx < _config.coherent_dms().size(); ++dm_idx)
    {
        dm_delays[dm_idx] = _dedispersers[dm_idx]->max_delay() * tsamp;
    } 
    _handler.init(header);
}

template <typename InputType, typename OutputType, typename Handler>
void IncoherentDedispersionPipeline<InputType, OutputType, Handler>::operator()(InputVectorType const& data, std::size_t dm_idx)
{
    _output_buffers[dm_idx].metalike(data);
    _agg_buffers[dm_idx]->push_back(data.vector());
}

} // namespace skyweaver