#ifndef TICK_INCLUDED
#define TICK_INCLUDED

#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <string>
#include <cstring>
#include <sstream>

struct Tick
{
    float price;

    Tick(std::string tick_data); //Constructor
};

Tick::Tick(std::string tick_data)
{
    int price_end;
    price_end = tick_data.find(",",27);
    price = atof(tick_data.substr(25,price_end-25).c_str());
}
#endif // TICK_INCLUDED
