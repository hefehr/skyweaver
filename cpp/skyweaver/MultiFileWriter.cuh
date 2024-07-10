#ifndef SKYWEAVER_MULTIFILEWRITER_CUH
#define SKYWEAVER_MULTIFILEWRITER_CUH

#include "skyweaver/DescribedVector.hpp"
#include "skyweaver/FileOutputStream.hpp"
#include "skyweaver/ObservationHeader.hpp"
#include "skyweaver/PipelineConfig.hpp"
#include "skyweaver/types.cuh"
#include "thrust/host_vector.h"
#include <map>

namespace skyweaver
{

template <typename VectorType>
class MultiFileWriter
{
    public:
        MultiFileWriter(PipelineConfig const& config, std::string tag="");
        MultiFileWriter(MultiFileWriter const&) = delete;
        ~MultiFileWriter();

        void init(ObservationHeader const& header);

        bool operator()(VectorType const& stream_data, std::size_t stream_idx = 0);

    private:
        bool has_stream(std::size_t stream_idx);
        FileStream& create_stream(VectorType const& stream_data, std::size_t stream_idx);
        std::string get_basefilename(VectorType const& stream_data, std::size_t stream_idx);
        std::string get_extension(VectorType const& stream_data);


        PipelineConfig const& _config;
        std::string _tag;
        ObservationHeader _header;
        std::map<std::size_t, std::unique_ptr<FileStream>> _file_streams;
        std::map<std::size_t, std::vector<std::size_t>> _stream_dims;
        std::vector<long double> _dm_delays;

};

} // namespace skyweaver

#include "skyweaver/detail/MultiFileWriter.cu"

#endif // SKYWEAVER_MULTIFILEWRITER_CUH