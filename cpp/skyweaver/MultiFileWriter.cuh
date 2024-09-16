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
/**
 * @brief A class for handling writing of DescribedVectors
 *
 */
template <typename VectorType>
class MultiFileWriter
{
  public:
    /**
     * @brief Construct a new Multi File Writer object
     *
     * @param config  The pipeline configuration
     * @param tag     A string tag to be added to the file name
     *                (used to avoid clashing file names).
     */
    MultiFileWriter(PipelineConfig const& config, std::string tag = "");
    MultiFileWriter(MultiFileWriter const&) = delete;

    /**
     * @brief Destroy the Multi File Writer object
     *
     */
    ~MultiFileWriter();

    /**
     * @brief Initialise the writer
     *
     * @param header The observation header for the current observation
     */
    void init(ObservationHeader const& header);

    /**
     * @brief Write data to file(s)
     *
     * @param stream_data A DescribedVector instance of streams
     * @param stream_idx  The index of the stream being written to
     * @return true
     * @return false
     *
     * @details The MultiFileWriter supports an arbitrary number of output
     * streams. If the given stream index does not already have an output stream
     *          one will be created. Stream indexes do not need to be
     * contiguous.
     *
     * @note The usecase for stream indexes is to allow different output files
     * for different DedispersionPlanBlocks (i.e. different outputs per coherent
     *       DM).
     */
    bool operator()(VectorType const& stream_data, std::size_t stream_idx = 0);

  private:
    bool has_stream(std::size_t stream_idx);
    FileOutputStream& create_stream(VectorType const& stream_data,
                              std::size_t stream_idx);
    std::string get_output_dir(VectorType const& stream_data,
                               std::size_t stream_idx);
    std::string get_basefilename(VectorType const& stream_data,
                                 std::size_t stream_idx);
    std::string get_extension(VectorType const& stream_data);
    void wait_for_space();
    PipelineConfig const& _config;
    std::string _tag;
    ObservationHeader _header;
    std::map<std::size_t, std::unique_ptr<FileOutputStream>> _file_streams;
    std::map<std::size_t, std::vector<std::size_t>> _stream_dims;
    std::vector<long double> _dm_delays;
};

} // namespace skyweaver

#include "skyweaver/detail/MultiFileWriter.cu"

#endif // SKYWEAVER_MULTIFILEWRITER_CUH