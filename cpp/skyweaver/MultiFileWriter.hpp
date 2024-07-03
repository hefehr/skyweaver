#ifndef SKYWEAVER_MULTIFILEWRITER_HPP
#define SKYWEAVER_MULTIFILEWRITER_HPP

#include "skyweaver/FileOutputStream.hpp"
#include "skyweaver/ObservationHeader.hpp"
#include "skyweaver/PipelineConfig.hpp"
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
    bool operator()(VectorType const& btf_powers, std::size_t dm_idx);

  private:
    void make_dada_header() const;

    PipelineConfig const& _config;
    std::string _tag;
    std::vector<std::unique_ptr<FileStream>> _file_streams;
    std::vector<long double> _dm_delays;
};

} // namespace skyweaver

#include "skyweaver/detail/MultiFileWriter.cpp"

#endif // SKYWEAVER_MULTIBEAMFILEWRITER_HPP