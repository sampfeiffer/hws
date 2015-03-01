#ifndef COUNTERPARTY_INCLUDED
#define COUNTERPARTY_INCLUDED

#include <cmath>
#include "test_fx.h"

struct Counterparty{
    int cp_id;
    float hazard_rate;
    double cva;
    thrust::device_vector<Fx> fx_deals;

    Counterparty(int cp_id_, float hazard_rate_); //Constructor
    void print();
    double prob_default(int t);
};

// Constructor
Counterparty::Counterparty(int cp_id_, float hazard_rate_)
{
    cp_id = cp_id_;
    hazard_rate = hazard_rate_;
    cva = 0;
}


void Counterparty::print()
{
    std::cout << "Counterparty id: " << cp_id
              << "\nHazard rate " << hazard_rate << "\n";
}

double Counterparty::prob_default(int t)
{
    return std::exp(-hazard_rate*(t-1)/360.0) - std::exp(-hazard_rate*t/360.0);
}

#endif // COUNTERPARTY_INCLUDED
