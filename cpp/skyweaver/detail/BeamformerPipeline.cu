#include "ascii_header.h"
#include "cuda.h"
#include "psrdada_cpp/Header.hpp"
#include "psrdada_cpp/cuda_utils.hpp"
#include "skyweaver/BeamformerPipeline.cuh"

#include <cstdlib>
#include <exception>
#include <stdexcept>

namespace skyweaver
{

class NullHandler
{
  public:
    template <typename... Args>
    void init(HeaderType const& header) {};

    template <typename... Args>
    bool operator()(Args... args)
    {
        return false;
    };
};

template <typename CBHandler, typename IBHandler, typename StatsHandler>
BeamformerPipeline<CBHandler, IBHandler, StatsHandler>::BeamformerPipeline(
    PipelineConfig const& config,
    _CBHandler& cb_handler,
    _IBHandler& ib_handler,
    _StatsHandler& stats_handler) _: _config(config),
                                     _cb_handler(cb_handler),
                                     _ib_handler(ib_handler),
                                     _ _stats_handler(stats_handler) _
{
    BOOST_LOG_TRIVIAL(debug) << "Allocating buffers from config sizes";
    std::size_t nsamples = _config.gulp_length_samps();
    if(nsamples % _config.nsamples_per_heap() != 0) {
        throw std::runtime_error("Gulp size is not a multiple of "
                                 "the number of samples per heap");
    }
    std::size_t nheap_groups = nsamples / _config.nsamples_per_heap();
    std::size_t input_taftp_size =
        nheap_groups * nsamples / _config.nsamples_per_heap();
    _taftp_from_host.resize(input_taftp_size, {0, 0});

    std::size_t expected_cb_size =
        (_config.nbeams() * nsamples / _config.cb_tscrunch() *
         _config.nchans() / _config.cb_fscrunch());
    _btf_cbs.resize(expected_cb_size, 0);

    std::size_t expected_ib_size = (nsamples / _config.ib_tscrunch() *
                                    _config.nchans() / _config.ib_fscrunch());
    _tf_ib.resize(expected_ib_size, 0);

    // Calculate the timestamp step per block
    _sample_clock_tick_per_block = 2 * _config.total_nchans() * nsamples;
    BOOST_LOG_TRIVIAL(debug)
        << "Sample clock tick per block: " << _sample_clock_tick_per_block;

    BOOST_LOG_TRIVIAL(debug) << "Allocating CUDA streams";
    CUDA_ERROR_CHECK(cudaStreamCreate(&_h2d_copy_stream));
    CUDA_ERROR_CHECK(cudaStreamCreate(&_processing_stream));
    CUDA_ERROR_CHECK(cudaStreamCreate(&_d2h_copy_stream));

    BOOST_LOG_TRIVIAL(debug) << "Constructing delay and weights managers";
    _delay_manager.reset(new DelayManager(_config, _h2d_copy_stream));
    _weights_manager.reset(new WeightsManager(_config, _processing_stream));
    _stats_manager.reset(new StatisticsCalculator(_config, _processing_stream));
    _split_transpose.reset(new Transposer(_config));
    _coherent_beamformer.reset(new CoherentBeamformer(_config));
    _incoherent_beamformer.reset(new IncoherentBeamformer(_config));
}

template <typename CBHandler, typename IBHandler, typename StatsHandler>
BeamformerPipeline<CBHandler, IBHandler, StatsHandler>::~BeamformerPipeline()
{
    CUDA_ERROR_CHECK(cudaStreamDestroy(_h2d_copy_stream));
    CUDA_ERROR_CHECK(cudaStreamDestroy(_processing_stream));
    CUDA_ERROR_CHECK(cudaStreamDestroy(_d2h_copy_stream));
}

template <typename CBHandler, typename IBHandler>
BeamformerPipeline<CBHandler, IBHandler, StatsHandler>::init(
    ObservationHeader const& header)
{
    BOOST_LOG_TRIVIAL(debug) << "Initialising beamformer pipeline";
    _header = header;
    _cb_handler.init(_header);
    _ib_handler.init(_header);
    _stats_handler.init(_header);
}

template <typename CBHandler, typename IBHandler, typename StatsHandler>
void BeamformerPipeline<CBHandler, IBHandler, StatsHandler>::process(
    VoltageVectorType& taftp_vec,
    PowerVectorType& tbtf_vec,
    PowerVectorType& tf_vec, )
{
    BOOST_LOG_TRIVIAL(debug) << "Executing beamforming pipeline";

    // Need to add the unix timestmap to the delay manager here
    // to fetch valid delays for this epoch.
    BOOST_LOG_TRIVIAL(debug) << "Checking for delay updates";
    auto const& delays = _delay_manager->delays(_unix_timestamp);

    // Stays the same
    BOOST_LOG_TRIVIAL(debug)
        << "Calculating weights at unix time: " << _unix_timestamp;
    auto const& weights = _weights_manager->weights(delays,
                                                    _unix_timestamp,
                                                    _delay_manager->epoch());

    BOOST_LOG_TRIVIAL(debug)
        << "Transposing input data from TAFTP to FTPA order";
    _split_transpose->transpose(taftp_vec,
                                _split_transpose_output,
                                _processing_stream);

    // Stays the same
    BOOST_LOG_TRIVIAL(debug) << "Checking if channel statistics update request";
    _stats_manager->calculate_statistics(ftpa_voltages, _processing_stream);

    _dispenser->digest(ftpa_voltages);

    for(auto dm: _config.coherent_dms()) {
        for(unsigned int freq_idx = 0; freq_idx < _config.nchans();
            ++freq_idx) {
            _dispenser->dispense(tpa_voltages, freq_idx);
            _coherent_dedisperser->dedisperse(tpa_voltages,
                                              ftpa_dm_voltages,
                                              freq_idx * tpa_size,
                                              dm);

            // Not yet implemented
            auto const& ib_scalings = _stats_manager->ib_scalings();
            auto const& ib_offsets  = _stats_manager->ib_offsets();
            _incoherent_beamformer->beamform(ftpa_dm_voltages,
                                             tf_raw_vec,
                                             tf_vec,
                                             ib_scaling,
                                             ib_offsets,
                                             _processing_stream);

            auto const& cb_scalings = _stats_manager->cb_scalings();
            auto const& cb_offsets  = _stats_manager->cb_offsets();
            _coherent_beamformer->beamform(ftpa_dm_voltages,
                                           weights,
                                           cb_scalings,
                                           cb_offsets,
                                           tf_raw_vec,
                                           _processing_stream);

            _ _cb_handler(_btf_cbs);
            _ _ib_handler(_tf_ib);
            _ _stats_handler(_stats_manager.statistics());
        }
    }
}
}

template <typename CBHandler, typename IBHandler, typename StatsHandler>
bool BeamformerPipeline<CBHandler, IBHandler, StatsHandler>::operator()(
    thrust::host_vector<char2> const& taftp_on_host)
{
    if(taftp_on_host.size() != _taftp_from_host.size()) {
        throw std::runtime_error(
            std::string("Unexpected buffer size, expected ") +
            std::to_string(taftp_on_host.size()) + " but got " +
            std::to_string(_taftp_from_host.size()));
    }
    CUDA_ERROR_CHECK(cudaMemcpyAsync(
        static_cast<void*>(thrust::raw_pointer_cast(_taftp_from_host.data())),
        static_cast<void*>(thrust::raw_pointer_cast(taftp_on_host.data())),
        taftp_on_host.size() * sizeof(char2),
        cudaMemcpyHostToDevice,
        _h2d_copy_stream));
    CUDA_ERROR_CHECK(cudaStreamSynchronize(_h2d_copy_stream));

    // Calculate the unix timestamp for the block that is about to be processed
    // (which is the block passed the last time that operator() was called)
    _unix_timestamp =
        _sync_time +
        (_sample_clock_start + _sample_clock_tick_per_block / _sample_clock);

    process(_taftp_from_host, _btf_cbs, _tf_ib);
    CUDA_ERROR_CHECK(cudaStreamSynchronize(_processing_stream));
    CUDA_ERROR_CHECK(cudaStreamSynchronize(_d2h_copy_stream));
    return false;
}

} // namespace skyweaver