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
    static __thread char* start_date;
    static __thread double start_time;
    static __thread float start_price;
    static __thread int start_units;
    static __thread int counter;
    static int bad_counter;
    static std::vector<int> bad_vector;

    char date[9];
    double time;
    float price;
    int units;

    Tick(std::string tick_data); //Constructor
    //void static_init();
    void print();
    void print2();
    int check_data(bool update_ref=true);
    std::string print3();
};

__thread char* Tick::start_date;
__thread double Tick::start_time;
__thread float Tick::start_price;
__thread int Tick::start_units;
__thread int Tick::counter;
int Tick::bad_counter = 0;
std::vector<int> Tick::bad_vector(16,0);


Tick::Tick(std::string tick_data)
{
    int price_end;

    strcpy(date, tick_data.substr(0,8).c_str());

    time = 0;
    time += atoi(tick_data.substr(9,2).c_str())*3600;
    time += atoi(tick_data.substr(12,2).c_str())*60;
    time += atof(tick_data.substr(15,9).c_str());

    price_end = tick_data.find(",",27);
    price = atof(tick_data.substr(25,price_end-25).c_str());

    units = atoi(tick_data.substr(price_end+1,tick_data.find("\n",34)-price_end+1).c_str());

    ++counter;
    //if (!start_time) static_init();

}

//void Tick::static_init()
//{
//    start_date = {'\0'};
//    start_time = 0;
//    start_price = 0;
//    start_units = 0;
//}

void Tick::print()
{
    std::cout << "date: " << date << "\n";
    std::cout << std::setprecision(12) << "time: " << time << "\n";
    std::cout << std::setprecision(7) << "price: " << price << "\n";
    std::cout << "units: " << units << "\n";
}

void Tick::print2()
{
    std::cout << counter << " " << date << ":" << time << "," << price << "," << units << " " << start_price;
}

int Tick::check_data(bool update_ref)
{
    int error_num=0;
    // Check date.
    if (std::string(date) != std::string(start_date)) error_num+=1;

    // Check time
    if (time-start_time > 2) error_num+=2;

    // Check price
    if (std::abs((price-start_price)/start_price) > 0.6 || price <= 0) error_num+=4;

    // Check units
    if (units < 0) error_num+=8;

    if (error_num){
        ++bad_counter;
        ++bad_vector[error_num];
        //print2();
        //std::cout << " " << error_num << " " << start_date << "\n";
    }

    // If data is good. Update static data.
    else if (counter%100 == 0 && update_ref){
        start_time = time;
        start_price = price;
        start_units = units;  //is this needed??
    }

    return error_num;
}

std::string Tick::print3()
{
    //if (time<10) std::cout <<time<< "\n";
    std::stringstream text;
    double integral;
    double seconds_frac = modf(time, &integral);

    int time_int = int(time);

    int seconds_int = time_int % 60;
    int minutes = (time_int-seconds_int)/60 % 60;
    int hours = (time_int-minutes*60-seconds_int) / 3600;

    text << date << ":";

    text << std::setfill('0') << std::setw(2) << hours << ":"
         << std::setfill('0') << std::setw(2) << minutes << ":"
         << std::setfill('0') << std::setw(8) << seconds_int + seconds_frac;
    text << "," << price << "," << units;
    return text.str();
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
