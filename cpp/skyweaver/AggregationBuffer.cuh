#ifndef SKYWEAVER_AGGREGATIONBUFFER_CUH
#define SKYWEAVER_AGGREGATIONBUFFER_CUH

#include "thrust/host_vector.h"
#include <functional>
#include <algorithm>

namespace skyweaver {

/**


    AggregationBuffer<float> agg([](std::vector<float> const& x){
        std::cout << "DISPATCH" << "\n";
    }, 10, 0, 1);
    
    std::vector<float> z(65);
    
    agg.push_back(z);
    agg.push_back(z);

    BTF --> TBF


    For each coherent DM we have one agg buffer and one dedisperser

    IncoherentDedisperser<Handler> idedisperser([](std::vector<T>& tb_buffer){
        ??? --> write to file maybe?
    }); 
    
    AggregationBuffer<T> agg_buffer(
        [&](std::vector<float>& tfb_buffer)
        {
            idedisperser.dedisperse(tfb_buffer); --> will call the handler with t - max_delay output samples;
        }
    )

    Usage is:

    agg_buffer.push_back(tfb_data);
    agg_buffer.push_back(tfb_data);
    ...
    agg_buffer.push_back(tfb_data);
    --> output handler called, buffer reset

    // Where is threading done?
    - Each dedisperser owns its own thread maybe?
    - Subsequent calls must join the previous thread if active an block until ready
    - Or just use OpenMP to parallelise the individual calls

    

*/


template <typename T>
class AggregationBuffer
{

public:
    typedef thrust::host_vector<T> BufferType;
    typedef std::function<void(BufferType const&)> DispatchCallback;

public:
    AggregationBuffer(DispatchCallback callback, std::size_t dispatch_size, 
                      std::size_t overlap_size=0, std::size_t slot_size=1);
    ~AggregationBuffer();
    AggregationBuffer(AggregationBuffer const&) = delete;
    AggregationBuffer& operator=(AggregationBuffer const&) = delete;

    template<template <typename, typename> class Container, typename A>
    void push_back(Container<T, A> const&);

private:

    template<template <typename, typename> class Container, typename A>
    void push_back(typename Container<T, A>::const_iterator begin, std::size_t size);
    void dispatch();
    std::size_t remaining_slots() const;
    void reset();

    DispatchCallback _callback;
    std::size_t _dispatch_size;
    std::size_t _overlap_size;
    std::size_t _slot_size;
    BufferType _buffer;
    typename BufferType::iterator _buffer_iter;
};

}

#include "skyweaver/detail/AggregationBuffer.cu"

#endif //SKYWEAVER_AGGREGATIONBUFFER_CUH

