#ifndef TICK_INCLUDED
#define TICK_INCLUDED

#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>

struct Tick
{
    thread_local static std::string start_date;
    thread_local static double start_time;
    thread_local static float start_price;
    thread_local static int start_units;
    thread_local static int counter;
    static int bad_counter;
    static std::vector<int> bad_vector;

    std::string date;
    double time;
    float price;
    int units;

    Tick(std::string tick_data, bool is_start_data=false); //Constructor
    void print();
    void print2();
    int check_data();
};

// This will need to be changed
thread_local std::string Tick::start_date = "";
thread_local double Tick::start_time = 0;
thread_local float Tick::start_price = 0;
thread_local int Tick::start_units = 0;
thread_local int Tick::counter = 0;
int Tick::bad_counter = 0;
std::vector<int> Tick::bad_vector(16,0);


Tick::Tick(std::string tick_data, bool is_start_data)
{
    int price_end;

    date = tick_data.substr(0,8);

    time = 0;
    time += atoi(tick_data.substr(9,2).c_str())*3600;
    time += atoi(tick_data.substr(12,2).c_str())*60;
    time += atof(tick_data.substr(15,9).c_str());

    price_end = tick_data.find(",",27);
    price = atof(tick_data.substr(25,price_end-25).c_str());

    units = atoi(tick_data.substr(price_end+1,tick_data.find("\n",34)-price_end+1).c_str());

    if (!is_start_data) ++counter;

}

void Tick::print()
{
    std::cout << "date: " << date << "\n";
    std::cout << std::setprecision(12) << "time: " << time << "\n";
    std::cout << std::setprecision(7) << "price: " << price << "\n";
    std::cout << "units: " << units << "\n";
}

void Tick::print2()
{
    std::cout << date << ":" << time << "," << price << "," << units << " " << start_price << "\n";
}

int Tick::check_data()
{
    int error_num=0;
    // Check date.
    if (date != start_date) error_num+=1;

    // Check time
    if (time-start_time > 2) error_num+=2;

    // Check price
    if (std::abs((price-start_price)/start_price) > 0.6 || price <= 0) error_num+=4;

    // Check units
    if (units < 0) error_num+=8;

    if (error_num){
        ++bad_counter;
        ++bad_vector[error_num];
    }

    // If data is good. Update static data.
    else if (counter%100 == 0){
        start_time = time;
        start_price = price;
        start_units = units;  //is this needed??
    }

    return error_num;
}

bool compare_date(const Tick &obj1, const Tick &obj2)
{
    return (obj1.date < obj2.date);
}

bool compare_time(const Tick &obj1, const Tick &obj2)
{
    return (obj1.time < obj2.time);
}

bool compare_price(const Tick &obj1, const Tick &obj2)
{
    return (obj1.price < obj2.price);
}

bool compare_units(const Tick &obj1, const Tick &obj2)
{
    return (obj1.units < obj2.units);
}

#endif // TICK_INCLUDED
