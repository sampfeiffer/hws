#ifndef SCRUB_TICK_INCLUDED
#define SCRUB_TICK_INCLUDED

#include <iostream>
#include <iomanip>
#include <cmath>
#include <string>
#include <cstring>
#include <sstream>

// Struct for tick data.

struct Tick
{
    // Reference values to compare data to for checking if the data is bad.
    static __thread char* start_date;
    static __thread double start_time;
    static __thread float start_price;
    static __thread int start_units;

    static __thread int counter; //total number of ticks in thread
    static int bad_counter; //total number of bad data

    // Member variables
    char date[9];
    double time;
    float price;
    int units;

    // Member functions
    Tick(std::string tick_data, bool start_info=false); //Constructor
    int check_data(bool update_ref=true);
    std::string print();
};

__thread char* Tick::start_date;
__thread double Tick::start_time;
__thread float Tick::start_price;
__thread int Tick::start_units;
__thread int Tick::counter;
int Tick::bad_counter = 0;


// Constructor. Gets data from input string.
Tick::Tick(std::string tick_data, bool start_info)
{
    int price_end;

    strcpy(date, tick_data.substr(0,8).c_str()); //Get date

    //Get time
    time = 0;
    time += atoi(tick_data.substr(9,2).c_str())*3600;
    time += atoi(tick_data.substr(12,2).c_str())*60;
    time += atof(tick_data.substr(15,9).c_str());

    //Get price
    price_end = tick_data.find(",",27);
    price = atof(tick_data.substr(25,price_end-25).c_str());

    //Get units
    units = atoi(tick_data.substr(price_end+1,tick_data.find("\n",34)-price_end+1).c_str());

    if (!start_info) ++counter;
}

// Checks if the data is bad.
// If update_ref is false, don't update the reference values.
int Tick::check_data(bool update_ref)
{
    int error_num=0;

    // Check date.
    if (std::string(date) != std::string(start_date)) error_num+=1;

    // Check time
    if (std::abs(time-start_time) > 2) error_num+=2;

    // Check for unrealistic price jumps
    if (std::abs((price-start_price)/(start_price*(time-start_time))) > 5 || price <= 0) error_num+=4;

    // Check units
    if (units <= 0 || units > 5000*long(start_units)) error_num+=8;

    if (error_num) ++bad_counter;

    // If data is good. Update static data.
    else if (update_ref){
        start_time = time;
        start_price = price;
        start_units = units;
    }

    return error_num;
}

// Returns a string of the data in the same format as the input file.
std::string Tick::print()
{
    std::stringstream text;

    //Convert time to original format.
    double integral;
    double seconds_frac = modf(time, &integral);
    int time_int = int(time);
    int seconds_int = time_int % 60;
    int minutes = (time_int-seconds_int)/60 % 60;
    int hours = (time_int-minutes*60-seconds_int) / 3600;

    text << date << ":"
         << std::setfill('0') << std::setw(2) << hours << ":"
         << std::setfill('0') << std::setw(2) << minutes << ":"
         << std::setfill('0') << std::setw(8) << seconds_int + seconds_frac
         << "," << price
         << "," << units;
    return text.str();
}

// The following four functions are used for the sort algorithm to decide
// what characteristic of the Tick to sort by.
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

#endif // SCRUB_TICK_INCLUDED
