#ifndef SWAP_INCLUDED
#define SWAP_INCLUDED

#include <iostream>
#include <string>
#include <sstream>

struct Swap{
    int swap_id, notional;
    char denomination, position;
    float fixed_rate;

    Swap(std::string deal_text); //Constructor
    void print();

};

// Constructor
Swap::Swap(std::string deal_text)
{
    std::stringstream deal_text_ss(deal_text);
    deal_text_ss >> swap_id;
    deal_text_ss >> denomination;
    deal_text_ss >> notional;
    deal_text_ss >> fixed_rate;
    deal_text_ss >> position;
}

void Swap::print()
{
    std::cout << "\nswap_id " << swap_id
              << "\ndenomination " << denomination
              << "\nnotional " << notional
              << "\nfixed_rate " << fixed_rate
              << "\nposition " << position << "\n";
}

#endif // SWAP_INCLUDED
