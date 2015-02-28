#ifndef FX_INCLUDED
#define FX_INCLUDED

#include <iostream>
#include <string>
#include <sstream>

struct Fx{
    int fx_id, notional;
    char position;

    Fx(std::string deal_text); //Constructor
    void print();
    void print_short();
    float value(double fx_rate_beg, double fx_rate_cur);

};

// Constructor
Fx::Fx(std::string deal_text)
{
    std::stringstream deal_text_ss(deal_text);
    deal_text_ss >> fx_id;
    deal_text_ss >> notional;
    deal_text_ss >> position;
}

void Fx::print()
{
    std::cout << "\nfx_id " << fx_id
              << "\nnotional " << notional
              << "\nposition " << position << "\n";
}

void Fx::print_short()
{
    std::cout << "    " << fx_id << " " << notional << " " << position << "\n";
}

float Fx::value(double fx_rate_beg, double fx_rate_cur)
{
    int sign=1;
    if (position == 's') sign=-1;

    return sign*notional*(std::max(fx_rate_cur,0.0)/fx_rate_beg - 1);
}

#endif // FX_INCLUDED
