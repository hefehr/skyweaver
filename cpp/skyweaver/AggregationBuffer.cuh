#ifndef SKYWEAVER_AGGREGATIONBUFFER_CUH
#define SKYWEAVER_AGGREGATIONBUFFER_CUH

#include "skyweaver/DescribedVector.hpp"
#include "thrust/host_vector.h"

#include <algorithm>
#include <functional>
#include <thrust/mr/allocator.h>
#include <thrust/mr/universal_memory_resource.h>

namespace skyweaver
{

/**
 * @brief A class for handling buffering of time series
 *
 * @tparam T The data type being buffered
 *
 * @details The buffers are implemented on host RAM but can be pushed to
 *          using device_vectors.
 */
template <typename T>
class AggregationBuffer
{
  public:
    typedef thrust::host_vector<T, PinnedAllocator<T>> BufferTypeH;
    typedef std::function<void(BufferTypeH const&)> DispatchCallback;

  public:
    /**
     * @brief Construct a new Aggregation Buffer object
     *
     * @param callback       A callback to be executed on each release of the
     * buffer
     * @param dispatch_size  The number of samples in each release (minus the
     * overlap) in units of the batch size
     * @param overlap_size   The overlap between subsequent blocks of data in
     * units of the batch size
     * @param batch_size      The batch size
     */
    AggregationBuffer(DispatchCallback callback,
                      std::size_t dispatch_size,
                      std::size_t overlap_size = 0,
                      std::size_t batch_size   = 1);

    /**
     * @brief Destroy the Aggregation Buffer object
     *
     */
    ~AggregationBuffer();

    AggregationBuffer(AggregationBuffer const&)            = delete;
    AggregationBuffer& operator=(AggregationBuffer const&) = delete;

    /**
     * @brief Push a block of data into the buffer
     *
     * @tparam Container The template type of the source container
     * @tparam A         The allocator type of the source container
     */
    template <template <typename, typename> class Container, typename A>
    void push_back(Container<T, A> const&);

  private:
    template <template <typename, typename> class Container, typename A>
    void push_back(typename Container<T, A>::const_iterator begin,
                   std::size_t size);
    void dispatch();
    std::size_t remaining_slots() const;
    void reset();

    DispatchCallback _callback;
    std::size_t _dispatch_size;
    std::size_t _overlap_size;
    std::size_t _batch_size;
    BufferTypeH _buffer;
    typename BufferTypeH::iterator _buffer_iter;
};

} // namespace skyweaver

#include "skyweaver/detail/AggregationBuffer.cu"

#endif // SKYWEAVER_AGGREGATIONBUFFER_CUH
