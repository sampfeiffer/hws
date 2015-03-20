#ifndef TIMING_INCLUDED
#define TIMING_INCLUDED

#include <iostream>
#include <string>
#include <sstream>
#include <chrono>

// Handles timing the program
struct Timing
{
    std::string program_part;
    std::chrono::steady_clock::time_point start_time;
    float elapsed;

    Timing(std::string program_part_); //Constructor
    void end_timing();
    std::string print();
};

//Constructor
Timing::Timing(std::string program_part_)
{
    program_part = program_part_;
    start_time = std::chrono::steady_clock::now();
}

// Calculate the time elapsed since the start_time
void Timing::end_timing()
{
    elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now()-start_time).count()/1000.0;
}

// Return the time elapsed
std::string Timing::print()
{
    std::stringstream ss;
    ss << "Time elapsed - " << program_part << ": " << elapsed << " seconds\n";
    return ss.str();
}

#endif // TIMING_INCLUDED
