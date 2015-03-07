#ifndef COUNTERPARTY_INCLUDED
#define COUNTERPARTY_INCLUDED

#include <cmath>
#include "fx2.h"
#include "swap2.h"

struct Counterparty{
    int cp_id, num_of_fx, num_of_swap;
    float hazard_rate;

    Counterparty(){};
    Counterparty(int cp_id_, float hazard_rate_, int fx_count, int swap_count); //Constructor
    __device__ __host__ float prob_default(const int t);
};

// Constructor
Counterparty::Counterparty(int cp_id_, float hazard_rate_, int fx_count, int swap_count)
{
    cp_id = cp_id_;
    hazard_rate = hazard_rate_;
    num_of_fx = fx_count;
    num_of_swap = swap_count;
}

__device__ __host__
float Counterparty::prob_default(const int t)
{
    return std::exp(-hazard_rate*(t-1)/float(360.0)) - std::exp(-hazard_rate*t/float(360.0));
}

#endif // COUNTERPARTY_INCLUDED
