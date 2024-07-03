#ifndef SKYWEAVER_MULTIFILEWRITER_CUH
#define SKYWEAVER_MULTIFILEWRITER_CUH

#include "skyweaver/FileOutputStream.hpp"
#include "skyweaver/ObservationHeader.hpp"
#include "skyweaver/PipelineConfig.hpp"
#include "skyweaver/types.cuh"
#include "thrust/host_vector.h"

namespace skyweaver
{

class MultiFileWriter
{
  public:
    MultiFileWriter(PipelineConfig const& config, std::string tag="");
    MultiFileWriter(MultiFileWriter const&) = delete;
    ~MultiFileWriter();

    void init(ObservationHeader const& header);
    void init(ObservationHeader const& header, std::vector<long double> const& dm_delays);

    template <typename VectorType>
    typename std::enable_if<!is_device_vector<VectorType>::value, bool>::type 
    operator()(VectorType const& btf_powers, std::size_t dm_idx);

    template <typename VectorType>
    typename std::enable_if<is_device_vector<VectorType>::value, bool>::type 
    operator()(VectorType const& btf_powers, std::size_t dm_idx);

  private:
    void make_dada_header() const;

    PipelineConfig const& _config;
    std::string _tag;
    std::vector<std::unique_ptr<FileStream>> _file_streams;
    std::vector<long double> _dm_delays;
};

} // namespace skyweaver

#include "skyweaver/detail/MultiFileWriter.cu"

#endif // SKYWEAVER_MULTIFILEWRITER_CUH