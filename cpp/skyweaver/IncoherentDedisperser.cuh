#ifndef SKYWEAVER_INCOHERENTDEDISPERSER_CUH
#define SKYWEAVER_INCOHERENTDEDISPERSER_CUH

#include "skyweaver/PipelineConfig.hpp"

namespace skyweaver {

/**

New plan...

Incoherent dedispersion with be done on the host system

There will be one thread per coherent/incoherent DM. 

Should accept multiple incoherent DMs per coherent DM but the principal 
use case it to compute one DM per coherent DM.

Should output one file per DM in TB order (beams on inner dimension)
File writing should be handled by downstream system

Data from the GPU will come in TBTF order

First implement brute force method.

These should be async with respect to the handler

*/

class IncoherentDedisperser
{
public:
    IncoherentDedisperser(PipelineConfig const& config, std::vector<float> const& dms);
    IncoherentDedisperser(PipelineConfig const& config, float dm);
    ~IncoherentDedisperser();
    IncoherentDedisperser(IncoherentDedisperser const&) = delete;
    IncoherentDedisperser& operator=(IncoherentDedisperser const&) = delete;

    template <typename InputVectorType, typename OutputVectorType>
    void dedisperse(InputVectorType const& tfb_powers,   
                    OutputVectorType& dtb_powers);
    std::vector<int> const& delays() const;
    int max_delay() const;

private:
    void prepare();


    PipelineConfig const& _config;
    std::vector<float> _dms;
    int _max_delay;
    std::vector<int> _delays;
};


} // namespace skyweaver

#endif //SKYWEAVER_INCOHERENTDEDISPERSER_CUH