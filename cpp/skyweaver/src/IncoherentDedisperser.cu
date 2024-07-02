#include "skyweaver/IncoherentDedisperser.cuh"
#include "skyweaver/types.cuh"
#include "thrust/host_vector.h"

namespace {
    #define DM_CONSTANT 4.148806423e15 // Hz^2 pc^-1 cm^3 s
}

namespace skyweaver {

/**
Functor for calculating channel delays.
Note all frequencies are Hz.
*/
struct DMDelay
{
    double _dm;
    double _f_ref; // reference frequency Hz
    double _tsamp; // sampling interval seconds

    __host__ __device__
    DMDelay(double dm, double f_ref, double tsamp)
    : _dm(dm), _f_ref(f_ref), _tsamp(tsamp){}

    __host__ __device__
    int operator()(double const& freq){
        return static_cast<int>((DM_CONSTANT * (1/(_f_ref * _f_ref) - 1/(freq * freq)) * _dm) / _tsamp + 0.5);
    }
};

IncoherentDedisperser::IncoherentDedisperser(PipelineConfig const& config, 
                                             std::vector<float> const& dms)
: _config(config)
, _dms(dms)
, _delays(dms.size() * config.channel_frequencies().size())
, _max_delay(0)
, _scale_factor(1.0f)
{
    prepare();
}

IncoherentDedisperser::IncoherentDedisperser(PipelineConfig const& config, 
                                             float dm)
: IncoherentDedisperser(config, std::vector<float>(1, dm))
{
}

IncoherentDedisperser::~IncoherentDedisperser()
{
}

void IncoherentDedisperser::prepare()
{
    std::size_t nchans = _config.channel_frequencies().size();
    double chbw = _config.bandwidth() / _config.nchans();
    double tsamp = _config.cb_tscrunch() / chbw;
    for (int dm_idx = 0; dm_idx < _dms.size(); ++dm_idx)
    {
        std::transform(
            _config.channel_frequencies().begin(),
             _config.channel_frequencies().end(),
            _delays.begin() + nchans * dm_idx,
            DMDelay(_dms[dm_idx], _config.channel_frequencies().front(), tsamp)
        );
    }
    auto it = std::max_element(_delays.begin(), _delays.end());
    _max_delay = *it;
    _scale_factor = std::sqrt(_config.nchans());
}

std::vector<int> const& IncoherentDedisperser::delays() const
{
    return _delays;
}

int IncoherentDedisperser::max_delay() const
{
    return _max_delay;
}

// Incoherent beamformer should have an option to output 8-bit data.

template <typename InputVectorType, typename OutputVectorType>
void IncoherentDedisperser::dedisperse<InputVectorType, OutputVectorType>(
    InputVectorType const& tfb_powers, OutputVectorType& tdb_powers)
{
    typedef typename value_traits<typename OutputVectorType::value_type>::promoted_type AccumulatorType;
    typedef std::vector<AccumulatorType> AccumulatorVectorType;

    const std::size_t nchans   = _config.channel_frequencies().size();
    const std::size_t nbeams   = _config.nbeams();
    const std::size_t ndms     = _dms.size();
    const std::size_t nsamples = tfb_powers.size() / (nbeams * nchans);
    if (nsamples <= _max_delay)
    {
        throw std::runtime_error("Fewer than max_delay samples passed to dedisperse method");
    }
    const std::size_t bf       = nbeams * nchans;
    tdb_powers.resize((nsamples - _max_delay) * nbeams * ndms);
    AccumulatorVectorType powers(nbeams);
    for (int t_idx = 0; t_idx < (nsamples - _max_delay); ++t_idx)
    {
        int t_output_offset = t_idx * nbeams * ndms;
        for (int dm_idx = 0; dm_idx < ndms; ++dm_idx)
        {
            int offset = nchans * dm_idx;
            std::fill(powers.begin(), powers.end(), value_traits<typename decltype(powers)::value_type>::zero());
            for (int f_idx = 0; f_idx < nchans; ++f_idx)
            {
                int idx = (t_idx + _delays[offset + f_idx]) * bf + f_idx * nbeams;
                for (int b_idx = 0; b_idx < nbeams; ++b_idx)
                {
                    powers[b_idx] += tfb_powers[idx + b_idx];
                }
            }
            int output_offset = t_output_offset + dm_idx * nbeams;
            std::transform(powers.begin(), powers.end(), tdb_powers.begin() + output_offset, 
                           [this](AccumulatorType const& value){
                               return clamp<typename OutputVectorType::value_type>(value / _scale_factor);
                           });
        }
    }
}

// This is the set of explicitly supported template arguments
template void IncoherentDedisperser::dedisperse<thrust::host_vector<int8_t>, thrust::host_vector<int8_t>>(
    thrust::host_vector<int8_t> const& tfb_powers, thrust::host_vector<int8_t>& tdb_powers);
template void IncoherentDedisperser::dedisperse<thrust::host_vector<char4>, thrust::host_vector<char4>>(
    thrust::host_vector<char4> const& tfb_powers, thrust::host_vector<char4>& tdb_powers);

} // namespace skyweaver