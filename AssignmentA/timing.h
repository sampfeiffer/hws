#ifndef TIMING_INCLUDED
#define TIMING_INCLUDED

#include <iostream>
#include <string>
#include <sstream>
#include <chrono>

// Handles timing the program
struct Timing
{
    std::chrono::steady_clock::time_point start_time;
    float elapsed;

    void start_timing();
    void end_timing();
    std::string print(std::string event);
};

// Save the start time.
void Timing::start_timing()
{
    start_time = std::chrono::steady_clock::now();
}

// Calculate the time elapsed since the start_time
void Timing::end_timing()
{
    elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now()-start_time).count()/1000.0;
}

// Return the time elapsed
std::string Timing::print(std::string event)
{
    std::stringstream ss;
    ss << "Time elapsed - " << event << ": " << elapsed << " seconds\n";
    return ss.str();
}

#endif // TIMING_INCLUDED
