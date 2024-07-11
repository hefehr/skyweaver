#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace skyweaver
{
/**
 * @brief A helper class to provide stopwatch functionality
 * 
 */
class Timer
{
  public:
    /**
     * @brief Start the stopwatch for the given name
     * 
     * @param name The name of the timed scope
     */
    void start(const std::string& name);

    /**
     * @brief Stop the stopwatch for the given name
     * 
     * @param name The name of the timed scope
     */
    void stop(const std::string& name);

    /**
     * @brief Show the timings for the given name
     * 
     * @param name The name of the timed scope
     */
    void show_timings(const std::string& name) const;

    /**
     * @brief Show the timings for all named scopes
     * 
     */
    void show_all_timings() const;

  private:
    std::unordered_map<std::string,
                       std::chrono::high_resolution_clock::time_point>
        _startTimes;
    std::unordered_map<std::string, std::vector<long long>> _timingData;
};

} // namespace skyweaver