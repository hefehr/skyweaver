#include "boost/log/trivial.hpp"
#include "skyweaver/AggregationBuffer.cuh"
#include "thrust/copy.h"

#include <algorithm>
#include <iostream>
#include <iterator>

namespace skyweaver
{

namespace
{
template <typename VectorType>
void print_buffer(VectorType const& vec)
{
    std::cout << "buffer: ";
    for(auto const& a: vec) { std::cout << a << ","; }
    std::cout << "\n";
}
} // namespace

template <typename T>
AggregationBuffer<T>::AggregationBuffer(DispatchCallback callback,
                                        std::size_t dispatch_size,
                                        std::size_t overlap_size,
                                        std::size_t batch_size)
    : _callback(callback), _dispatch_size(dispatch_size),
      _overlap_size(overlap_size), _slot_size(batch_size)
{
    _buffer.resize((_dispatch_size + _overlap_size) * _slot_size, T{});
    reset();
}

template <typename T>
AggregationBuffer<T>::~AggregationBuffer()
{
}

template <typename T>
template <template <typename, typename> class Container, typename A>
void AggregationBuffer<T>::push_back(Container<T, A> const& data)
{
    push_back<Container, A>(data.cbegin(), data.size());
}

template <typename T>
template <template <typename, typename> class Container, typename A>
void AggregationBuffer<T>::push_back(
    typename Container<T, A>::const_iterator begin,
    std::size_t size)
{
    if(size % _slot_size != 0) {
        throw std::runtime_error(
            "input size is not a multiple of the slot size");
    }
    std::size_t nslots_to_copy = size / _slot_size;
    // BOOST_LOG_TRIVIAL(debug) << "Agg: nslots_to_copy = " << nslots_to_copy;
    while(nslots_to_copy > 0) {
        std::size_t rslots = remaining_slots();
        // BOOST_LOG_TRIVIAL(debug) << "Agg: remaining_slots = " << rslots;
        if(nslots_to_copy <= rslots) {
            // BOOST_LOG_TRIVIAL(debug) << "Agg: filling slots";
            _buffer_iter = thrust::copy(begin,
                                        begin + (nslots_to_copy * _slot_size),
                                        _buffer_iter);
            if(nslots_to_copy == rslots) {
                // BOOST_LOG_TRIVIAL(debug) << "Agg: slots copied, dispatching";
                dispatch();
            }
            return;
        } else {
            // BOOST_LOG_TRIVIAL(debug) << "Agg: filling all slots";
            std::size_t copy_size =
                static_cast<std::size_t>(rslots) * _slot_size;
            _buffer_iter = thrust::copy(begin, begin + copy_size, _buffer_iter);
            // BOOST_LOG_TRIVIAL(debug) << "Agg: all slots full, dispatching";
            dispatch();
            begin += copy_size;
            nslots_to_copy -= rslots;
            continue;
        }
    }
}

template <typename T>
void AggregationBuffer<T>::dispatch()
{
    _callback(_buffer);
    reset();
    if(_overlap_size > 0) {
        _buffer_iter = std::copy(_buffer.end() - (_overlap_size * _slot_size),
                                 _buffer.end(),
                                 _buffer_iter);
    }
}

template <typename T>
std::size_t AggregationBuffer<T>::remaining_slots() const
{
    typename BufferType::const_iterator iter = _buffer_iter;
    typename BufferType::difference_type rslots =
        std::distance(iter, _buffer.cend()) / _slot_size;
    if(rslots < 0) {
        throw std::runtime_error("Iterator beyond end of buffer");
    }
    return static_cast<std::size_t>(rslots);
}

template <typename T>
void AggregationBuffer<T>::reset()
{
    _buffer_iter = _buffer.begin();
}

} // namespace skyweaver