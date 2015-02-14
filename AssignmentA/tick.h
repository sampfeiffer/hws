#ifndef TICK_INCLUDED
#define TICK_INCLUDED

#include <iostream>
#include <iomanip>
#include <cmath>

struct Tick
{
    static std::string start_date;
    static double start_time;
    static float start_price;
    static int start_units;
    static int counter;
    static int bad_counter;

    std::string date;
    double time;
    float price;
    int units;

    Tick(std::string tick_data, bool new_thread=false); //Constructor
    void print();
    void print2();
    int check_data();
};

// This will need to be changed
std::string Tick::start_date = "20140804";
double Tick::start_time = 36000.574914;
float Tick::start_price = 1173.56;
int Tick::start_units = 471577;
int Tick::counter = 0;
int Tick::bad_counter = 0;


Tick::Tick(std::string tick_data, bool new_thread)
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

    ++counter;

    if (new_thread){
        ;
    }

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
    std::cout << date << ":" << time << "," << price << "," << units;
}

int Tick::check_data()
{
    int error_num=0;
    // Check date.
    if (date != start_date) error_num+=1;

    // Check time
    if (time-start_time > 2) error_num+=2;

    // Check price
    //if (std::abs((price-start_price)/start_price) > 0.6 || price <= 0) error_num+=4;
    if (std::abs((price-start_price)/start_price) > 2 || price <= 0) error_num+=4;

    // Check units
    if (units < 0) error_num+=8;

    if (error_num){
        ++bad_counter;
        print2();
        std::cout << " " << error_num << "\n";
    }

    // If data is good. Update static data.
    else if (counter%100 == 0){
        start_time = time;
        start_price = price;
        start_units = units;  //is this needed??
    }

    return error_num;
}

#endif // TICK_INCLUDED
