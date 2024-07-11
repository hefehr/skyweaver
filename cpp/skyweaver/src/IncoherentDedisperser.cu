#include "skyweaver/DescribedVector.hpp"
#include "skyweaver/IncoherentDedisperser.cuh"
#include "skyweaver/dedispersion_utils.cuh"
#include "skyweaver/types.cuh"
#include "thrust/host_vector.h"

namespace
{
#define DM_CONSTANT 4.148806423e15 // Hz^2 pc^-1 cm^3 s
} // namespace

namespace skyweaver
{

IncoherentDedisperser::IncoherentDedisperser(PipelineConfig const& config,
                                             std::vector<float> const& dms,
                                             std::size_t tscrunch)
    : _config(config), _dms(dms), _tscrunch(tscrunch),
      _delays(dms.size() * config.channel_frequencies().size()), _max_delay(0),
      _scale_factor(1.0f)
{
    prepare();
}

IncoherentDedisperser::IncoherentDedisperser(PipelineConfig const& config,
                                             float dm,
                                             std::size_t tscrunch)
    : IncoherentDedisperser(config, std::vector<float>(1, dm), tscrunch)
{
}

IncoherentDedisperser::~IncoherentDedisperser()
{
}

void IncoherentDedisperser::prepare()
{
    std::size_t nchans = _config.channel_frequencies().size();
    double chbw        = _config.bandwidth() / _config.nchans();
    double tsamp       = _config.cb_tscrunch() / chbw;
    for(int dm_idx = 0; dm_idx < _dms.size(); ++dm_idx) {
        std::transform(_config.channel_frequencies().begin(),
                       _config.channel_frequencies().end(),
                       _delays.begin() + nchans * dm_idx,
                       DMSampleDelay(_dms[dm_idx],
                                     _config.channel_frequencies().front(),
                                     tsamp));
    }
    auto it       = std::max_element(_delays.begin(), _delays.end());
    _max_delay    = *it;
    _scale_factor = std::sqrt(_config.nchans() * _tscrunch);
}

std::vector<int> const& IncoherentDedisperser::delays() const
{
    return _delays;
}

int IncoherentDedisperser::max_delay() const
{
    return _max_delay;
}

template <typename InputVectorType, typename OutputVectorType>
void IncoherentDedisperser::dedisperse<InputVectorType, OutputVectorType>(
    InputVectorType const& tfb_powers,
    OutputVectorType& tdb_powers)
{
    typedef typename value_traits<
        typename OutputVectorType::value_type>::promoted_type AccumulatorType;
    typedef std::vector<AccumulatorType> AccumulatorVectorType;

    const std::size_t nchans   = _config.channel_frequencies().size();
    const std::size_t nbeams   = _config.nbeams();
    const std::size_t ndms     = _dms.size();
    const std::size_t nsamples = tfb_powers.size() / (nbeams * nchans);
    const std::size_t bf       = nbeams * nchans;
    if(nsamples <= _max_delay) {
        throw std::runtime_error(
            "Fewer than max_delay samples passed to dedisperse method");
    }
    if((nsamples - _max_delay) % _tscrunch != 0) {
        throw std::runtime_error(
            "(nsamples - max_delay) must be a multiple of tscrunch;");
    }
    tdb_powers.resize({(nsamples - _max_delay) / _tscrunch, ndms, nbeams});
    AccumulatorVectorType powers(nbeams);
    for(int t_idx = _max_delay; t_idx < nsamples; t_idx += _tscrunch) {
        int t_output_offset = (t_idx - _max_delay) / _tscrunch * nbeams * ndms;
        for(int dm_idx = 0; dm_idx < ndms; ++dm_idx) {
            int offset = nchans * dm_idx;
            std::fill(
                powers.begin(),
                powers.end(),
                value_traits<typename decltype(powers)::value_type>::zero());
            for(int tsub_idx = 0; tsub_idx < _tscrunch; ++tsub_idx) {
                for(int f_idx = 0; f_idx < nchans; ++f_idx) {
                    int idx =
                        ((t_idx + tsub_idx) - _delays[offset + f_idx]) * bf +
                        f_idx * nbeams;
                    for(int b_idx = 0; b_idx < nbeams; ++b_idx) {
                        powers[b_idx] += tfb_powers[idx + b_idx];
                    }
                }
                int output_offset = t_output_offset + dm_idx * nbeams;
                std::transform(
                    powers.begin(),
                    powers.end(),
                    tdb_powers.begin() + output_offset,
                    [this](AccumulatorType const& value) {
                        return clamp<typename OutputVectorType::value_type>(
                            value / _scale_factor);
                    });
            }
        }
    }
}

// This is the set of explicitly supported template arguments
template void IncoherentDedisperser::dedisperse<thrust::host_vector<int8_t>,
                                                TDBPowersH<int8_t>>(
    thrust::host_vector<int8_t> const& tfb_powers,
    TDBPowersH<int8_t>& tdb_powers);
template void IncoherentDedisperser::dedisperse<thrust::host_vector<char4>,
                                                TDBPowersH<char4>>(
    thrust::host_vector<char4> const& tfb_powers,
    TDBPowersH<char4>& tdb_powers);

} // namespace skyweaver