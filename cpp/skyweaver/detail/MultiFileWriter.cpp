#include "skyweaver/MultiFileWriter.hpp"

namespace skyweaver
{

template <typename VectorType>
bool MultiFileWriter::operator()(VectorType const& btf_powers,
                                 std::size_t dm_idx)
{
    _file_streams[dm_idx]->write(
        reinterpret_cast<char const*>(btf_powers.data()),
        btf_powers.size() * sizeof(typename VectorType::value_type));
    return false;
}

} // namespace name
