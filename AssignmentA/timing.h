#ifndef TIMING_INCLUDED
#define TIMING_INCLUDED

#include <iostream>
#include <chrono>

struct Timing
{
    std::chrono::steady_clock::time_point start_time, end_time;
    float elapsed;

    void start_timing();
    void end_timing();
    void print(std::string event);
};

void Timing::start_timing()
{
    start_time = std::chrono::steady_clock::now();
}

void Timing::end_timing()
{
    end_time = std::chrono::steady_clock::now();
    elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end_time-start_time).count()/1000.0;
}

void Timing::print(std::string event)
{
    std::cout << "Time elapsed - " << event << ": " << elapsed << " seconds\n";
}

#endif // TIMING_INCLUDED
