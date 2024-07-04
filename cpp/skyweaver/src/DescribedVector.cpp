#include "skyweaver/DescribedVector.hpp"
#include <sstream>

namespace skyweaver {

std::string dimensions_to_string(std::vector<Dimension> const& dims)
{
    std::stringstream dimstring;
    for (auto const& dim: dims)
    {
        switch(dim)
        {
            case TimeDim: 
                dimstring << "T";
                break;
            case FreqDim: 
                dimstring << "F";
                break;
            case BeamDim: 
                dimstring << "B";
                break;
            case AntennaDim: 
                dimstring << "A";
                break;
            case PolnDim: 
                dimstring << "P";
                break;
            case DispersionDim:
                dimstring << "D";
                break;
        }
    }
    return dimstring.str();
}

} // namespace skyweaver