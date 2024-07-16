#include "skyweaver/SigprocHeader.hpp"

#include <boost/algorithm/string.hpp>
#include <chrono>
#include <iterator>
#include <string>

namespace skyweaver
{

SigprocHeader::SigprocHeader()
{
}

SigprocHeader::~SigprocHeader()
{
}

void SigprocHeader::header_write(std::ostream& stream, std::string const& str)
{
    int len = str.size();
    stream.write(reinterpret_cast<char*>(&len), sizeof(int));
    stream << str;
}

void SigprocHeader::header_write(std::ostream& stream,
                                 std::string const& str,
                                 std::string const& name)
{
    header_write(stream, str);
    header_write(stream, name);
}

std::ostream& SigprocHeader::write_header(std::ostream& stream,
                                          FilHead const& params)
{
    header_write(stream, "HEADER_START");
    header_write<std::uint32_t>(stream, "telescope_id", params.telescopeid);
    header_write<std::uint32_t>(stream, "machine_id", params.machineid);
    header_write<std::uint32_t>(stream, "data_type", params.datatype);
    header_write<std::uint32_t>(stream, "barycentric", params.barycentric);
    header_write(stream, "source_name", params.source);
    header_write<double>(stream, "src_raj", params.ra);
    header_write<double>(stream, "src_dej", params.dec);
    header_write<std::uint32_t>(stream, "nbits", params.nbits);
    header_write<std::uint32_t>(stream, "nifs", params.nifs);
    header_write<std::uint32_t>(stream, "nchans", params.nchans);
    header_write<std::uint32_t>(stream, "ibeam", params.ibeam);
    header_write<double>(stream, "fch1", params.fch1);
    header_write<double>(stream, "foff", params.foff);
    header_write<double>(stream, "tstart", params.tstart);
    header_write<double>(stream, "tsamp", params.tsamp);
    header_write(stream, "HEADER_END");
    return stream;
}

} // namespace skyweaver
