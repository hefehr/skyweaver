#ifndef SKYWEAVER_DEDISPERSIONPLAN_HPP
#define SKYWEAVER_DEDISPERSIONPLAN_HPP

#include <iostream>
#include <vector>
#include <string>

namespace skyweaver {

struct DedispersionPlanBlock
{
    float coherent_dm;
    int tscrunch;
    std::vector<float> incoherent_dms;
};

class DedispersionPlan {
public:
    DedispersionPlan();
    ~DedispersionPlan();
    DedispersionPlan(const DedispersionPlan&) = delete;
    DedispersionPlan& operator=(const DedispersionPlan&) = delete;

    void add_block(float coherent_dm, float dm_begin, float dm_end, float dm_step, int tscrunch);
    void add_block(float coherent_dm, int tscrunch = 1);
    void add_block(DedispersionPlanBlock&& block);
    void add_block(std::string str);
    std::vector<float> const& coherent_dms() const;
    std::vector<DedispersionPlanBlock> const& blocks() const;
    DedispersionPlanBlock& operator[](std::size_t idx);
    DedispersionPlanBlock const & operator[](std::size_t idx) const;

private:
    std::vector<DedispersionPlanBlock> _blocks;
    std::vector<float> _coherent_dms;
};

std::ostream& operator<<(std::ostream& stream, DedispersionPlanBlock const& block);
std::ostream& operator<<(std::ostream& stream, DedispersionPlan const& plan);

} // namespace skyweaver

#endif //SKYWEAVER_DEDISPERSIONPLAN_HPP