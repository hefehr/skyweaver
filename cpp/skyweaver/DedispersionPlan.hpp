#ifndef SKYWEAVER_DEDISPERSIONPLAN_HPP
#define SKYWEAVER_DEDISPERSIONPLAN_HPP

#include <iostream>
#include <string>
#include <vector>

namespace skyweaver
{

struct DedispersionPlanBlock {
    float coherent_dm;
    int tscrunch;
    std::vector<float> incoherent_dms;
};

class DedispersionPlan
{
  public:
    /**
     * @brief Construct a new Dedispersion Plan object
     * 
     */
    DedispersionPlan();

    /**
     * @brief Destroy the Dedispersion Plan object
     * 
     */
    ~DedispersionPlan();
    DedispersionPlan(const DedispersionPlan&)            = delete;
    DedispersionPlan& operator=(const DedispersionPlan&) = delete;

    /**
     * @brief Add a dedispersion plan block
     * 
     * @param coherent_dm  The reference coherent DM for the block
     * @param dm_begin     The start DM for the incoherent DM range
     * @param dm_end       The end DM for the incoherent DM range
     * @param dm_step      The DM step size
     * @param tscrunch     The factor by which the data should be integrated in time
     * 
     * @details The incoherent DMs are specified absolutely, not with respect to the 
     *          coherent DM.
     */
    void add_block(float coherent_dm,
                   float dm_begin,
                   float dm_end,
                   float dm_step,
                   int tscrunch);
    
    /**
     * @brief Add a dedispersion plan block
     * 
     * @param coherent_dm The reference coherent DM for the block
     * @param tscrunch The factor by which the data should be integrated in time
     * 
     * @details Implicitly sets the incoherent DM to be the same as the coherent DM
     */
    void add_block(float coherent_dm, int tscrunch = 1);

    /**
     * @brief Add an existing dispersion plan block
     * 
     * @param block A dispersion plan block
     */
    void add_block(DedispersionPlanBlock&& block);

    /**
     * @brief Add a dispersion plan block based on a string definition
     * 
     * @param str A block definition string
     * 
     * @details
     *    Defines a dedispersion plan to be executed.
     *    Argument is colon-separated with no spaces.
     *    Parameters:
     *      <coherent_dm>:<start_incoherent_dm>:<end_incoherent_dm>:<dm_step>:<tscrunch>
     *    The tscrunch is defined relative to the beamformer output.
     *
     *    e.g "5.0:0.0:10.0:0.1:1"
     *    or
     *    "5.0:1"
     *
     */
    void add_block(std::string str);

    /**
     * @brief Get the coherent DMs from all the blocks
     * 
     * @return std::vector<float> const& 
     */
    std::vector<float> const& coherent_dms() const;

    /**
     * @brief Get the dedispersion plan blocks
     * 
     * @return std::vector<DedispersionPlanBlock> const& 
     */
    std::vector<DedispersionPlanBlock> const& blocks() const;

    /**
     * @brief Get a mutable referece to a specific dedispersion plan block
     * 
     * @param idx The block index
     * @return DedispersionPlanBlock& 
     */
    DedispersionPlanBlock& operator[](std::size_t idx);

    /**
     * @brief Get a const reference to a specific dedispersion plan block
     * 
     * @param idx 
     * @return DedispersionPlanBlock const& 
     */
    DedispersionPlanBlock const& operator[](std::size_t idx) const;

  private:
    std::vector<DedispersionPlanBlock> _blocks;
    std::vector<float> _coherent_dms;
};

// Output stream overloads for showing plan and blocks
std::ostream& operator<<(std::ostream& stream,
                         DedispersionPlanBlock const& block);
std::ostream& operator<<(std::ostream& stream, DedispersionPlan const& plan);

} // namespace skyweaver

#endif // SKYWEAVER_DEDISPERSIONPLAN_HPP