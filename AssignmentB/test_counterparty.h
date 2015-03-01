#ifndef COUNTERPARTY_INCLUDED
#define COUNTERPARTY_INCLUDED

#include <cmath>
#include "fx.h"
#include "swap.h"

struct Counterparty{
    int cp_id;
    float hazard_rate;
    double cva;
    //std::vector<Fx> fx_deals;
    //std::vector<Swap> swap_deals;
    thrust::device_vector<Fx> fx_deals;
    thrust::device_vector<Swap> swap_deals;

    Counterparty(int cp_id_, float hazard_rate_); //Constructor
    void add_fx(int fx_id_, int notional_, char position_);
    void add_swap(int swap_id_, char denomination_, int notional_, float fixed_rate_, int tenor_, int position_);
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

void Counterparty::add_fx(int fx_id_, int notional_, char position_)
{
    fx_deals.push_back(Fx(fx_id_, notional_, position_));
}

void Counterparty::add_swap(int swap_id_, char denomination_, int notional_, float fixed_rate_, int tenor_, int position_)
{
    swap_deals.push_back(Swap(swap_id_, denomination_, notional_, fixed_rate_, tenor_, position_));
}

void Counterparty::print()
{
    std::cout << "Counterparty id: " << cp_id
              << "\nHazard rate " << hazard_rate << "\n";

    std::cout << "FX deals:\n";
    for (unsigned int i=0; i<fx_deals.size(); ++i){
        fx_deals[i].print_short();
    }
    std::cout << "Swaps deals:\n";
    for (unsigned int i=0; i<swap_deals.size(); ++i){
        swap_deals[i].print_short();
    }
    std::cout << "\n";
}

double Counterparty::prob_default(int t)
{
    return std::exp(-hazard_rate*(t-1)/360.0) - std::exp(-hazard_rate*t/360.0);
}

#endif // COUNTERPARTY_INCLUDED
