#include "skyweaver/Timer.hpp"

namespace skyweaver
{

void Timer::start(const std::string& functionName)
{
    if(_startTimes.find(functionName) != _startTimes.end()) {
        throw std::runtime_error("Timer already started for function: " +
                                 functionName);
    }
    _startTimes[functionName] = std::chrono::high_resolution_clock::now();
}

void Timer::stop(const std::string& functionName)
{
    if(_startTimes.find(functionName) == _startTimes.end()) {
        throw std::runtime_error("Timer not started for function: " +
                                 functionName);
    }
    auto endTime   = std::chrono::high_resolution_clock::now();
    auto startTime = _startTimes[functionName];
    _startTimes.erase(functionName);

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
                        endTime - startTime)
                        .count();
    _timingData[functionName].push_back(duration);
}

void Timer::show_timings(const std::string& functionName) const
{
    if(_timingData.find(functionName) == _timingData.end()) {
        throw std::runtime_error("No timing data for function: " +
                                 functionName);
    }

    const auto& times = _timingData.at(functionName);
    if(times.empty()) {
        throw std::runtime_error("No recorded times for function: " +
                                 functionName);
    }

    double average =
        std::accumulate(times.begin(), times.end(), 0.0) / times.size();
    auto [minIt, maxIt] = std::minmax_element(times.begin(), times.end());

    std::cout << "Statistics for function '" << functionName
              << "':" << std::endl;
    std::cout << "  Average time: " << std::fixed << std::setprecision(3)
              << average << " us" << std::endl;
    std::cout << "  Minimum time: " << *minIt << " us" << std::endl;
    std::cout << "  Maximum time: " << *maxIt << " us" << std::endl;
    std::cout << "  Number of measurements: " << times.size() << std::endl;
}

void Timer::show_all_timings() const
{
    if(_timingData.empty()) {
        std::cout << "No timing data available." << std::endl;
        return;
    }

    for(const auto& [functionName, times]: _timingData) {
        show_timings(functionName);
        std::cout << std::endl;
    }
}

} // namespace skyweaver