#include "skyweaver/Header.hpp"
#include <sstream>
#include <iomanip>

namespace skyweaver 
{

template <template <typename, typename> class Container, typename T, typename A>
void Header::set(char const* key, Container<T, A> const& values, std::size_t precision)
{
    std::stringstream ss;
    ss << std::setprecision(precision);
    bool first = true;
    for (auto const& val: values)
    {
        if (first) {first = false;}
        else {ss << ",";}
        ss << val;
    }
    set<std::string>(key, ss.str());
}

} // namespace skyweaver 