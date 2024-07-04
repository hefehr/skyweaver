#include <iostream>
#include <chrono>
#include <unordered_map>
#include <string>
#include <vector>
#include <algorithm>
#include <iomanip>  
#include <thread>
#include <numeric>

namespace skyweaver 
{

class Timer {
public:
    void start(const std::string& functionName);
    void stop(const std::string& functionName);
    void show_timings(const std::string& functionName) const;
    void show_all_timings() const;

private:
    std::unordered_map<std::string, std::chrono::high_resolution_clock::time_point> _startTimes;
    std::unordered_map<std::string, std::vector<long long>> _timingData;
};

} // namespace skyweaver



/**
void exampleFunction1() {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
}

void exampleFunction2() {
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
}

void exampleFunction3() {
    std::this_thread::sleep_for(std::chrono::milliseconds(150));
}

int main() {
    Timer timer;

    timer.start("exampleFunction1");
    exampleFunction1();
    timer.stop("exampleFunction1");

    timer.start("exampleFunction2");
    exampleFunction2();
    timer.stop("exampleFunction2");

    timer.start("exampleFunction1");
    exampleFunction1();
    timer.stop("exampleFunction1");

    timer.start("exampleFunction3");
    exampleFunction3();
    timer.stop("exampleFunction3");

    // Print statistics for individual functions
    timer.show_timings("exampleFunction1");
    timer.show_timings("exampleFunction2");
    timer.show_timings("exampleFunction3");

    // Print statistics for all functions
    timer.show_all_timings();

    return 0;
}
*/