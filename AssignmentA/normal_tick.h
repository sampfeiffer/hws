#ifndef NORMAL_TICK_INCLUDED
#define NORMAL_TICK_INCLUDED

#include <string>
#include <cstring>

// Struct for the tick data. For the normality check, only the price is needed

struct Tick
{
    float price;

    Tick(std::string tick_data); //Constructor
};

// Constructor. Figures out the price from the input string.
Tick::Tick(std::string tick_data)
{
    int price_end;
    price_end = tick_data.find(",",27);
    price = atof(tick_data.substr(25,price_end-25).c_str());
}
#endif // NORMAL_TICK_INCLUDED
