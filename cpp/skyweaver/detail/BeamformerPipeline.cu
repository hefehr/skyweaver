#include "cuda.h"
#include "psrdada_cpp/cuda_utils.hpp"
#include "skyweaver/BeamformerPipeline.cuh"

#include <cstdlib>
#include <exception>
#include <stdexcept>
#include <string>

namespace skyweaver
{

namespace
{
template <typename VectorType>
void peek(VectorType const& vec, std::size_t n = 10)
{
    std::cout << "Peeking at vector of length " << vec.size() << "\n";
    std::size_t n_to_peek = std::min(n, vec.size());
    thrust::host_vector<typename VectorType::value_type> h(n_to_peek);
    thrust::copy(vec.begin(), vec.begin() + n_to_peek, h.begin());
    bool first = true;
    for(auto const& val: h) {
        if(!first) {
            std::cout << ", ";
        } else {
            first = false;
        }
        if constexpr(std::is_same_v<typename VectorType::value_type, int8_t>) {
            std::cout << static_cast<int>(val);
        } else {
            std::cout << val;
        }
    }
    std::cout << "\n";
}
} // namespace

template <typename CBHandler,
          typename IBHandler,
          typename StatsHandler,
          typename BeamformerTraits>
BeamformerPipeline<CBHandler, IBHandler, StatsHandler, BeamformerTraits>::
    BeamformerPipeline(PipelineConfig const& config,
                       CBHandler& cb_handler,
                       IBHandler& ib_handler,
                       StatsHandler& stats_handler)
    : _config(config), _nbeamsets(0), _cb_handler(cb_handler),
      _ib_handler(ib_handler), _stats_handler(stats_handler),
      _unix_timestamp(0.0), _call_count(0)
{
    BOOST_LOG_NAMED_SCOPE("BeamformerPipeline::BeamformerPipeline");
    BOOST_LOG_TRIVIAL(debug) << "Constructing beanmformer pipeline";
    std::size_t nsamples = _config.gulp_length_samps();
    BOOST_LOG_TRIVIAL(debug)
        << "Expected gulp size: " << nsamples << " (samples)";
    if(nsamples % _config.nsamples_per_heap() != 0) {
        throw std::runtime_error("Gulp size is not a multiple of "
                                 "the number of samples per heap");
    }
    // Calculate the timestamp step per block
    _sample_clock_tick_per_block = 2 * _config.total_nchans() * nsamples;
    BOOST_LOG_TRIVIAL(debug)
        << "Sample clock tick per block: " << _sample_clock_tick_per_block;

    CUDA_ERROR_CHECK(cudaStreamCreate(&_h2d_copy_stream));
    CUDA_ERROR_CHECK(cudaStreamCreate(&_processing_stream));
    CUDA_ERROR_CHECK(cudaStreamCreate(&_d2h_copy_stream));

    float f_low  = _config.centre_frequency() - _config.bandwidth() / 2.0f;
    float f_high = _config.centre_frequency() + _config.bandwidth() / 2.0f;
    float tsamp  = _config.nchans() / _config.bandwidth();
    auto it      = std::max_element(_config.coherent_dms().begin(),
                               _config.coherent_dms().end());
    float max_dm = *it;
    BOOST_LOG_TRIVIAL(debug) << "Constructing coherent dedisperser plan";
    float max_dm_delay =
        CoherentDedisperser::get_dm_delay(f_low, f_high, max_dm);
    CoherentDedisperser::createConfig(_dedisperser_config,
                                      _config.gulp_length_samps(),
                                      max_dm_delay,
                                      _config.nchans(),
                                      _config.npol(),
                                      _config.nantennas(),
                                      tsamp,
                                      f_low,
                                      _config.bandwidth(),
                                      _config.coherent_dms());

    BOOST_LOG_TRIVIAL(debug) << "Constructing delay and weights managers";
    _delay_manager.reset(new DelayManager(_config, _h2d_copy_stream));
    _weights_manager.reset(new WeightsManager(_config, _processing_stream));
    _stats_manager.reset(new StatisticsCalculator(_config, _processing_stream));
    _transposer.reset(new Transposer(_config));
    _coherent_beamformer.reset(new CoherentBeamformer(_config));
    _coherent_dedisperser.reset(new CoherentDedisperser(_dedisperser_config));
    _incoherent_beamformer.reset(new IncoherentBeamformer(_config));
    _dispenser.reset(new BufferedDispenser(_config, _processing_stream));
    _nbeamsets = _delay_manager->nbeamsets();
    BOOST_LOG_TRIVIAL(debug)
        << "Delay model contains " << _nbeamsets << " beamsets";
}

template <typename CBHandler,
          typename IBHandler,
          typename StatsHandler,
          typename BeamformerTraits>
BeamformerPipeline<CBHandler, IBHandler, StatsHandler, BeamformerTraits>::
    ~BeamformerPipeline()
{
    BOOST_LOG_NAMED_SCOPE("BeamformerPipeline::~BeamformerPipeline");
    CUDA_ERROR_CHECK(cudaStreamDestroy(_h2d_copy_stream));
    CUDA_ERROR_CHECK(cudaStreamDestroy(_processing_stream));
    CUDA_ERROR_CHECK(cudaStreamDestroy(_d2h_copy_stream));
    _timer.show_all_timings();
}

template <typename CBHandler,
          typename IBHandler,
          typename StatsHandler,
          typename BeamformerTraits>
void BeamformerPipeline<CBHandler, IBHandler, StatsHandler, BeamformerTraits>::
    init(ObservationHeader const& header)
{
    BOOST_LOG_NAMED_SCOPE("BeamformerPipeline::init");
    BOOST_LOG_TRIVIAL(debug) << "Initialising beamformer pipeline";
    _header = header;
    _cb_handler.init(_header);
    _ib_handler.init(_header);
    _stats_handler.init(_header);
}

template <typename CBHandler,
          typename IBHandler,
          typename StatsHandler,
          typename BeamformerTraits>
void BeamformerPipeline<CBHandler, IBHandler, StatsHandler, BeamformerTraits>::
    process()
{
    BOOST_LOG_NAMED_SCOPE("BeamformerPipeline::process");

    BOOST_LOG_TRIVIAL(debug) << "Executing beamforming pipeline";

    // Need to add the unix timestmap to the delay manager here
    // to fetch valid delays for this epoch.
    BOOST_LOG_TRIVIAL(debug) << "Checking for delay updates";

    _timer.start("fetch delays");
    auto const& delays = _delay_manager->delays(_unix_timestamp);
    _timer.stop("fetch delays");

    // Stays the same
    BOOST_LOG_TRIVIAL(debug)
        << "Calculating weights at unix time: " << _unix_timestamp;

    _timer.start("calculate weights");
    auto const& weights = _weights_manager->weights(delays,
                                                    _unix_timestamp,
                                                    _delay_manager->epoch());
    _timer.stop("calculate weights");
    // BOOST_LOG_TRIVIAL(info) << "Peeking weights at epoch " <<
    // std::setprecision(15) <<  _unix_timestamp; peek(weights, weights.size());

    BOOST_LOG_TRIVIAL(debug)
        << "Transposing input data from TAFTP to FTPA order";
    _timer.start("transpose TAFTP to FTPA");
    _transposer->transpose(_taftp_from_host,
                           _ftpa_post_transpose,
                           _header.nantennas,
                           _processing_stream);
    _timer.stop("transpose TAFTP to FTPA");
    _ftpa_dedispersed.like(_ftpa_post_transpose);
    // Stays the same
    BOOST_LOG_TRIVIAL(debug) << "Checking if channel statistics update request";
    _timer.start("calculate statistics");
    _stats_manager->calculate_statistics(_ftpa_post_transpose);
    _timer.stop("calculate statistics");
    if(_call_count == 0) {
        _timer.start("update scalings");
        _stats_manager->update_scalings(_delay_manager->beamset_weights(),
                                        _delay_manager->nbeamsets());
        _timer.stop("update scalings");
    }

    // BOOST_LOG_TRIVIAL(debug) << "Peeking the statistics";
    // peek(_stats_manager->statistics(), 64);

    BOOST_LOG_TRIVIAL(debug)
        << "FTPA post transpose size: " << _ftpa_post_transpose.size();

    // BOOST_LOG_TRIVIAL(debug) << "peeking _ftpa_post_transpose";
    // peek(_ftpa_post_transpose);

    _timer.start("dispenser hoarding");
    _dispenser->hoard(_ftpa_post_transpose);
    _timer.stop("dispenser hoarding");

    for(unsigned int dm_idx = 0; dm_idx < _config.coherent_dms().size();
        ++dm_idx) {
        _timer.start("coherent dedispersion");
        for(unsigned int freq_idx = 0; freq_idx < _config.nchans();
            ++freq_idx) {
            auto const& tpa_voltages = _dispenser->dispense(freq_idx);
            _coherent_dedisperser->dedisperse(tpa_voltages,
                                              _ftpa_dedispersed,
                                              freq_idx,
                                              dm_idx);
        }
        _timer.stop("coherent dedispersion");

        // BOOST_LOG_TRIVIAL(debug) << "peeking _ftpa_dedispersed";
        // peek(_ftpa_dedispersed);
        _timer.start("incoherent beamforming");
        _incoherent_beamformer->beamform(_ftpa_dedispersed,
                                         _tf_ib_raw,
                                         _tf_ib,
                                         _stats_manager->ib_scaling(),
                                         _stats_manager->ib_offsets(),
                                         _delay_manager->beamset_weights(),
                                         _nbeamsets,
                                         _processing_stream);
        _timer.stop("incoherent beamforming");

        // BOOST_LOG_TRIVIAL(debug) << "peeking _tf_ib_raw";
        // peek(_tf_ib_raw);

        _timer.start("coherent beamforming");
        _coherent_beamformer->beamform(_ftpa_dedispersed,
                                       weights,
                                       _stats_manager->cb_scaling(),
                                       _stats_manager->cb_offsets(),
                                       _delay_manager->beamset_mapping(),
                                       _tf_ib_raw,
                                       _btf_cbs,
                                       _nbeamsets,
                                       _processing_stream);
        _timer.stop("coherent beamforming");

        // BOOST_LOG_TRIVIAL(debug) << "peeking _btf_cbs";
        // peek(_btf_cbs);

        _timer.start("coherent beam handler");
        _cb_handler(_btf_cbs, dm_idx);
        _timer.stop("coherent beam handler");

        _timer.start("incoherent beam handler");
        _ib_handler(_tf_ib, dm_idx);
        _timer.stop("incoherent beam handler");
    }
    _timer.start("statistics handler");
    _stats_handler(_stats_manager->statistics());
    _timer.stop("statistics handler");
}

template <typename CBHandler,
          typename IBHandler,
          typename StatsHandler,
          typename BeamformerTraits>
bool BeamformerPipeline<CBHandler, IBHandler, StatsHandler, BeamformerTraits>::
operator()(HostVoltageVectorType const& taftp_on_host)
{
    BOOST_LOG_NAMED_SCOPE("BeamformerPipeline::operator()");
    BOOST_LOG_TRIVIAL(debug) << "Pipeline operator() called with data: \n"
                             << taftp_on_host.describe();

    _taftp_from_host.like(taftp_on_host);

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
