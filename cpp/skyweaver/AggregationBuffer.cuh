#ifndef SKYWEAVER_AGGREGATIONBUFFER_CUH
#define SKYWEAVER_AGGREGATIONBUFFER_CUH

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

*/


template <typename T>
class AggregationBuffer
{

public:
    typedef std::function<void(std::vector<T> const&)> DispatchCallback;

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
    std::vector<T> _buffer;
    typename std::vector<T>::iterator _buffer_iter;
};

}

#include "skyweaver/detail/AggregationBuffer.cu"

#endif //SKYWEAVER_AGGREGATIONBUFFER_CUH

