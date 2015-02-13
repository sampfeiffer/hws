#ifndef TICK_INCLUDED
#define TICK_INCLUDED

#include <iostream>
#include <iomanip>

struct Tick
{
    std::string date;
    double time;
    float price;
    int units;

    Tick(std::string tick_data);
    void print();
};

Tick::Tick(std::string tick_data)
{
    date = tick_data.substr(0,8);
    time = 0;
    time += atoi(tick_data.substr(9,2).c_str())*3600;
    time += atoi(tick_data.substr(12,2).c_str())*60;
    time += atof(tick_data.substr(15,9).c_str());
    price = atof(tick_data.substr(25,tick_data.find(",",27)-25).c_str());
    units = atoi(tick_data.substr(33,tick_data.find("\n",34)-33).c_str());
}

void Tick::print()
{
    std::cout << "date: " << date << "\n";
    std::cout << std::setprecision(12) << "time: " << time << "\n";
    std::cout << std::setprecision(7) << "price: " << price << "\n";
    std::cout << "units: " << units << "\n";
}

#endif // TICK_INCLUDED
