#include "thrust/host_vector.h"

namespace skyweaver
{

template <typename VectorType>
typename std::enable_if<!is_device_vector<VectorType>::value, bool>::type 
MultiFileWriter::operator()(VectorType const& btf_powers,
                                 std::size_t dm_idx)
{
    _file_streams[dm_idx]->write(
        reinterpret_cast<char const*>(btf_powers.data()),
        btf_powers.size() * sizeof(typename VectorType::value_type));
    return false;
}

template <typename VectorType>
typename std::enable_if<is_device_vector<VectorType>::value, bool>::type 
MultiFileWriter::operator()(VectorType const& btf_powers,
                                 std::size_t dm_idx)
{
    thrust::host_vector<typename VectorType::value_type> btf_powers_h = btf_powers;
    return this->operator()(btf_powers_h, dm_idx);
}

} // namespace skyweaver
