#ifndef SKYWEAVER_MULTIFILEWRITER_HPP
#define SKYWEAVER_MULTIFILEWRITER_HPP

#include "skyweaver/ObservationHeader.hpp"
#include "skyweaver/PipelineConfig.hpp"
#include "skyweaver/FileOutputStream.hpp"
#include "thrust/host_vector.h"

namespace skyweaver
{

class MultiFileWriter
{
  public:
    
    typedef thrust::host_vector<int8_t> PowerVectorType;

    MultiFileWriter(PipelineConfig const& config);
    MultiFileWriter(MultiFileWriter const&) = delete;
    ~MultiFileWriter();

    void init(ObservationHeader const& header);
    
    bool operator()(PowerVectorType const& btf_powers, std::size_t dm_idx);

  private:
    void make_dada_header() const;

    PipelineConfig const& _config;
    std::vector<std::unique_ptr<FileStream>> _file_streams;
};

} // namespace skyweaver

#endif // SKYWEAVER_MULTIBEAMFILEWRITER_HPP