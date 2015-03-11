#ifndef TIMING_INCLUDED
#define TIMING_INCLUDED

#include <iostream>
#include <time.h>

struct Timing{
    clock_t start_time;
    std::string program_part;

    Timing(std::string program_part_);
    void end_timing();
};

Timing::Timing(std::string program_part_)
{
    start_time = clock();
    program_part = program_part_;
}

void Timing::end_timing()
{
    std::cout << "Timing: " << program_part << ": " << float(clock()-start_time)/CLOCKS_PER_SEC << " seconds.\n";
}

#endif // TIMING_INCLUDED
