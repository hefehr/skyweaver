#include "cuda.h"
#include "psrdada_cpp/cuda_utils.hpp"
#include "skyweaver/BeamformerPipeline.cuh"

#include <cstdlib>
#include <exception>
#include <stdexcept>
#include <string>

namespace skyweaver
{

template <typename CBHandler, typename IBHandler, typename StatsHandler, typename BeamformerTraits>
BeamformerPipeline<CBHandler, IBHandler, StatsHandler, BeamformerTraits>::BeamformerPipeline(
    PipelineConfig const& config,
    CBHandler& cb_handler,
    IBHandler& ib_handler,
    StatsHandler& stats_handler)
    : _config(config), _nbeamsets(0), _cb_handler(cb_handler),
      _ib_handler(ib_handler), _stats_handler(stats_handler),
      _unix_timestamp(0.0), _call_count(0)
{
    BOOST_LOG_TRIVIAL(debug) << "Constructing beanmformer pipeline";
    std::size_t nsamples = _config.gulp_length_samps();
    BOOST_LOG_TRIVIAL(debug)
        << "Expected gulp size: " << nsamples << " (samples)";
    if(nsamples % _config.nsamples_per_heap() != 0) {
        throw std::runtime_error("Gulp size is not a multiple of "
                                 "the number of samples per heap");
    }
    std::size_t nheap_groups = nsamples / _config.nsamples_per_heap();
    std::size_t input_taftp_size =
        nheap_groups * nsamples / _config.nsamples_per_heap();
    _taftp_from_host.resize(input_taftp_size, {0, 0});
    // Calculate the timestamp step per block
    _sample_clock_tick_per_block = 2 * _config.total_nchans() * nsamples;
    BOOST_LOG_TRIVIAL(debug)
        << "Sample clock tick per block: " << _sample_clock_tick_per_block;
    CUDA_ERROR_CHECK(cudaStreamCreate(&_h2d_copy_stream));
    CUDA_ERROR_CHECK(cudaStreamCreate(&_processing_stream));
    CUDA_ERROR_CHECK(cudaStreamCreate(&_d2h_copy_stream));

    BOOST_LOG_TRIVIAL(debug) << "Constructing delay and weights managers";
    _delay_manager.reset(new DelayManager(_config, _h2d_copy_stream));
    _weights_manager.reset(new WeightsManager(_config, _processing_stream));
    _stats_manager.reset(new StatisticsCalculator(_config, _processing_stream));
    _transposer.reset(new Transposer(_config));
    _coherent_beamformer.reset(new CoherentBeamformer(_config));
    _coherent_dedisperser.reset(
        new CoherentDedisperser(_config, _dedispeser_config));
    _incoherent_beamformer.reset(new IncoherentBeamformer(_config));
    _dispenser.reset(new BufferedDispenser(_config, _processing_stream));
    _nbeamsets = _delay_manager->nbeamsets();
    BOOST_LOG_TRIVIAL(debug)
        << "Delay model contains " << _nbeamsets << " beamsets";
}

template <typename CBHandler, typename IBHandler, typename StatsHandler, typename BeamformerTraits>
BeamformerPipeline<CBHandler, IBHandler, StatsHandler, BeamformerTraits>::~BeamformerPipeline()
{
    CUDA_ERROR_CHECK(cudaStreamDestroy(_h2d_copy_stream));
    CUDA_ERROR_CHECK(cudaStreamDestroy(_processing_stream));
    CUDA_ERROR_CHECK(cudaStreamDestroy(_d2h_copy_stream));
}

template <typename CBHandler, typename IBHandler, typename StatsHandler, typename BeamformerTraits>
void BeamformerPipeline<CBHandler, IBHandler, StatsHandler, BeamformerTraits>::init(
    ObservationHeader const& header)
{
    BOOST_LOG_TRIVIAL(debug) << "Initialising beamformer pipeline";
    _header = header;
    _cb_handler.init(_header);
    _ib_handler.init(_header);
    _stats_handler.init(_header);
    _taftp_from_host.resize(_config.gulp_length_samps() * header.nantennas *
                                _config.nchans() * _config.npol(),
                            {0, 0});
    BOOST_LOG_TRIVIAL(debug) << "Resized TAFTP input vector to "
                             << _taftp_from_host.size() << " elements";
}

template <typename CBHandler, typename IBHandler, typename StatsHandler, typename BeamformerTraits>
void BeamformerPipeline<CBHandler, IBHandler, StatsHandler, BeamformerTraits>::process()
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
    _transposer->transpose(_taftp_from_host,
                           _ftpa_post_transpose,
                           _header.nantennas,
                           _processing_stream);

    // Stays the same
    BOOST_LOG_TRIVIAL(debug) << "Checking if channel statistics update request";
    _stats_manager->calculate_statistics(_ftpa_post_transpose);

    BOOST_LOG_TRIVIAL(debug)
        << "FTPA post transpose size: " << _ftpa_post_transpose.size();

    _dispenser->hoard(_ftpa_post_transpose);

    for(unsigned int dm_idx = 0; dm_idx < _config.coherent_dms().size();
        ++dm_idx) {
        for(unsigned int freq_idx = 0; freq_idx < _config.nchans();
            ++freq_idx) {
            BOOST_LOG_TRIVIAL(debug) << "{{{[[[<<< DM Idx: " << dm_idx
                                     << " F Idx: " << freq_idx << " >>>]]]}}}";
            BOOST_LOG_TRIVIAL(debug) << "Dispensing some voltages";
            auto const& tpa_voltages = _dispenser->dispense(freq_idx);
            BOOST_LOG_TRIVIAL(debug) << "Attempting to segfault";

            _coherent_dedisperser->dedisperse(
                tpa_voltages,
                _ftpa_dedispersed,
                freq_idx * _ftpa_post_transpose.size() / _config.nchans(),
                dm_idx);
        }

        _incoherent_beamformer->beamform(_ftpa_dedispersed,
                                         _tf_ib_raw,
                                         _tf_ib,
                                         _stats_manager->ib_scaling(),
                                         _stats_manager->ib_offsets(),
                                         _delay_manager->beamset_weights(),
                                         _nbeamsets,
                                         _processing_stream);

        _coherent_beamformer->beamform(_ftpa_dedispersed,
                                       weights,
                                       _stats_manager->cb_scaling(),
                                       _stats_manager->cb_offsets(),
                                       _delay_manager->beamset_mapping(),
                                       _tf_ib_raw,
                                       _btf_cbs,
                                       _nbeamsets,
                                       _processing_stream);
        _cb_handler(_btf_cbs, dm_idx);
        _ib_handler(_tf_ib, dm_idx);
    }
    _stats_handler(_stats_manager->statistics());
}

template <typename CBHandler, typename IBHandler, typename StatsHandler, typename BeamformerTraits>
bool BeamformerPipeline<CBHandler, IBHandler, StatsHandler, BeamformerTraits>::operator()(
    HostVoltageVectorType const& taftp_on_host)
{
    BOOST_LOG_TRIVIAL(debug) << "Pipeline operator() called";
    BOOST_LOG_TRIVIAL(debug) << "taftp_on_host size: " << taftp_on_host.size();

    if(taftp_on_host.size() != _taftp_from_host.size()) {
        throw std::runtime_error(
            std::string("Unexpected buffer size, expected ") +
            std::to_string(taftp_on_host.size()) + " but got " +
            std::to_string(_taftp_from_host.size()));
    }

    CUDA_ERROR_CHECK(cudaMemcpyAsync(
        static_cast<void*>(thrust::raw_pointer_cast(_taftp_from_host.data())),
        static_cast<void const*>(
            thrust::raw_pointer_cast(taftp_on_host.data())),
        taftp_on_host.size() * sizeof(char2),
        cudaMemcpyHostToDevice,
        _h2d_copy_stream));
    CUDA_ERROR_CHECK(cudaStreamSynchronize(_h2d_copy_stream));

    // Calculate the unix timestamp for the block that is about to be
    // processed
    _unix_timestamp =
        _header.utc_start +
        static_cast<long double>(_call_count * _sample_clock_tick_per_block) /
            _header.sample_clock;
    process();
    CUDA_ERROR_CHECK(cudaStreamSynchronize(_processing_stream));
    CUDA_ERROR_CHECK(cudaStreamSynchronize(_d2h_copy_stream));
    ++_call_count;
    return false;
}
} // namespace skyweaver