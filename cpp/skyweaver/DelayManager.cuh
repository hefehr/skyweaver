#ifndef SKYWEAVER_DELAYMANAGER_CUH
#define SKYWEAVER_DELAYMANAGER_CUH

#include "skyweaver/PipelineConfig.hpp"

#include <boost/log/trivial.hpp>
#include <exception>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

namespace skyweaver
{

class InvalidDelayEpoch: public std::exception
{
  private:
    double epoch;

  public:
    InvalidDelayEpoch(double _epoch): epoch(_epoch) {}

    const char* what() const noexcept
    {
        std::ostringstream error_msg;
        error_msg << "No valid delay solution for unix epoch " << epoch;
        const char* cstr = error_msg.str().c_str();
        return cstr;
    }
};

/**
 * @brief A struct wrapping the header of each delay model in the delay file
 *
 */
struct DelayModelHeader {
    // Number of beams in the model set
    uint32_t nbeams;
    // Number of antennas in the model set
    uint32_t nantennas;
    // The start of the validity of the model as a unix epoch
    double start_epoch;
    // The end of the validity of the model as a unix epoch
    double end_epoch;
};

typedef float3 DelayModel;

/**
 * @brief      Class for managing loading of delay models from
 *             file and provisioning of them on the GPU.
 */
class DelayManager
{
  public:
    /**
     * These vectors are in (nbeam, nantenna) order, i.e. the 
     * fastest dimension is antenna. This is also how the vectors 
     * are store on file.
     */
    typedef thrust::device_vector<DelayModel> DelayVectorDType;
    typedef thrust::host_vector<DelayModel> DelayVectorHType;

  public:
    /**
     * @brief Construct a new Delay Manager object
     *
     * @param delay_file A file containing delay models in skyweaver format
     * @param stream A cuda stream on which to execute host to device copies
     */
    DelayManager(PipelineConfig const& config, cudaStream_t stream);
    ~DelayManager();
    DelayManager(DelayManager const&) = delete;

    /**
     * @brief Get the delay model for the given epoch
     *
     * @param epoch A Unix epoch for which to fetch delays.
     *
     * @details Implemented for strictly increasing epochs.
     *
     * @return A device vector containing the current delays
     */
    DelayVectorDType const& delays(double epoch);

    /**
     * @brief Return the UNIX epoch of the current delay model
     * 
     * @return double UNIX epoch
     */
    double epoch() const;

  private:
    bool validate_model(double epoch) const;
    void read_next_model();
    void safe_read(char* buffer, std::size_t nbytes);

    PipelineConfig const& _config;
    cudaStream_t _copy_stream;
    DelayModelHeader _header;
    std::ifstream _input_stream;
    DelayVectorHType _delays_h;
    DelayVectorDType _delays_d;
};

} // namespace skyweaver

#endif // SKYWEAVER_DELAYMANAGER_CUH