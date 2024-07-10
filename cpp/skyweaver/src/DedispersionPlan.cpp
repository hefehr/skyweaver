#include "skyweaver/DedispersionPlan.hpp"
#include <sstream>
#include <iomanip>


namespace skyweaver {


DedispersionPlan::DedispersionPlan(){

}

DedispersionPlan::~DedispersionPlan(){

}

void DedispersionPlan::add_block(float coherent_dm, float dm_begin, float dm_end, float dm_step, int tscrunch)
{
    DedispersionPlanBlock block;
    block.coherent_dm = coherent_dm;
    block.tscrunch = tscrunch;
    for (float dm = dm_begin; dm < dm_end; dm += dm_step){
        block.incoherent_dms.push_back(dm);
    }
    if (block.incoherent_dms.empty())
    {
        block.incoherent_dms.push_back(dm_begin);
    }
    add_block(std::move(block));
}

void DedispersionPlan::add_block(float coherent_dm, int tscrunch)
{
    DedispersionPlanBlock block;
    block.coherent_dm = coherent_dm;
    block.tscrunch = tscrunch;
    block.incoherent_dms = {coherent_dm};
    add_block(std::move(block));
}

void DedispersionPlan::add_block(DedispersionPlanBlock&& block)
{
    _coherent_dms.push_back(block.coherent_dm);
    _blocks.emplace_back(std::move(block));
}

void DedispersionPlan::add_block(std::string str)
{   
    std::vector<std::string> parts;
    std::stringstream ss(str);
    std::string item;
    while (std::getline(ss, item, ':')) {
        parts.push_back(item);
    }
    try {
        switch (parts.size()){
            case 1:
                add_block(std::stof(parts[0]));
                break;
            case 2:
                add_block(std::stof(parts[0]), std::stoi(parts[1]));
                break;
            case 5:
                add_block(std::stof(parts[0]), std::stof(parts[1]), std::stof(parts[2]), std::stof(parts[3]), std::stoi(parts[4]));
                break;
            default:
                throw std::runtime_error(std::string("Block defition string of unexpected size: ") + str);
        }
    } catch (std::invalid_argument& e) {
        throw std::invalid_argument(std::string("Error parsing block definition string: ") + e.what());
    } catch (const std::out_of_range& e) {
        throw std::out_of_range("What are you doing?");
    }
}

std::vector<float> const& DedispersionPlan::coherent_dms() const
{
    return _coherent_dms;
}

std::vector<DedispersionPlanBlock> const& DedispersionPlan::blocks() const
{
    return _blocks;
}

std::ostream& operator<<(std::ostream& stream, DedispersionPlanBlock const& block)
{
    stream << "Coherent DM: " << block.coherent_dm << ", "
           << "Tscrunch: " << block.tscrunch << ", "
           << "nDMs: " << block.incoherent_dms.size() << ", "
           << "DMs: ";
    bool first = true;
    for (auto const& dm: block.incoherent_dms)
    {
        if (first){
            first = false;
        } else {
            stream << ", ";
        }
        stream << std::setprecision(5) << dm;
    }
    return stream;
}

std::ostream& operator<<(std::ostream& stream, DedispersionPlan const& plan)
{
    stream << "DedispersionPlan: \n";
    int ii = 0;
    for (auto const& block: plan.blocks())
    {
        stream << "  Block " << ii << " -> " << block << "\n";
        ++ii;
    }
    return stream;
}

} // namespace skyweaver