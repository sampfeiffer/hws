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

#endif // FX_INCLUDED
