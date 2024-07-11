#ifndef SKYWEAVER_TRANSPOSER_HPP
#define SKYWEAVER_TRANSPOSER_HPP

#include "psrdada_cpp/common.hpp"
#include "skyweaver/DescribedVector.hpp"
#include "skyweaver/PipelineConfig.hpp"
#include "thrust/device_vector.h"

namespace skyweaver
{
namespace kernel
{

/**
 * @brief      Perform a split transpose of input data
 *             in TAFTP order.
 *
 * @param      input            8-bit complex voltage data in TAFTP order
 * @param      output           8-bit complex voltage data in FTPA order
 * @param[in]  total_nantennas  The total number of antennas (e.g. T[A]FTP)
 * @param[in]  input_antennas   The number of antennas in the input data
 * @param[in]  output_antennas  The number of antennas to pad to in the output
 *                              (0 valued antennas used for padding)
 * @param[in]  nchans           The number of frequency channels (e.g. TA[F]TP)
 * @param[in]  ntimestamps      The number of timestamps (outer T dimension,
 * e.g. [T]AFTP)
 */
__global__ void split_transpose_k(char2 const* __restrict__ input,
                                  char2* __restrict__ output,
                                  int input_antennas,
                                  int output_antennas,
                                  int start_antenna,
                                  int nchans,
                                  int ntimestamps);

} // namespace kernel

/**
 * @brief      Class for split transposing voltage data
 *
 */
class Transposer
{
  public:
    typedef TAFTPVoltagesD<char2> InputVoltageType;
    typedef FTPAVoltagesD<char2> OutputVoltageType;

  public:
    /**
     * @brief      Create a new split transposer
     *
     * @param      config  The pipeline configuration
     */
    explicit Transposer(PipelineConfig const& config);
    ~Transposer();
    Transposer(Transposer const&) = delete;

    /**
     * @brief      Perform a split transpose on the data
     *
     * @param      taftp_voltages  The input TAFTP voltages
     * @param      ftpa_voltages   The output FTPA voltages
     * @param[in]  input_nantennas The number of antennas in the input data
     * @param[in]  stream          The cuda stream to use
     */
    void transpose(InputVoltageType const& taftp_voltages,
                   OutputVoltageType& ftpa_voltages,
                   std::size_t input_nantennas,
                   cudaStream_t stream);

  private:
    PipelineConfig const& _config;
};

} // namespace skyweaver

#endif // SKYWEAVER_TRANSPOSER_HPP