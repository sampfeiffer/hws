#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cstdlib>
#include <chrono>
#include <algorithm>
#include <thread>
#include <sstream>

// Needed to time things.
#define TIMER_OBJ std::chrono::steady_clock::time_point
#define NOW std::chrono::steady_clock::now()
#define ELAPSED_TIME std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count()


struct Tick{
    std::string date;
    float time;
    float price;
    int units;

    Data(std::string date_, float time_, float _price, int _units){
        date = date_
        time = time_;
        price = _price;
        units = _units;
    }
};

// Converts the time part of the date string to an float.
float time_to_float(std::string time)
{
    float num=0;
    num+=atoi(time.substr(9,2).c_str())*3600;
    num+=atoi(time.substr(12,2).c_str())*60;
    num+=atof(time.substr(15,9).c_str());
    return num;
}

// Checks if data is good.
bool is_data_good(std::string record, vector<Tick> &my_vector)
{
    Tick *last_elem = &my_vector[my_vector.size()-1];





    if (price <= std::max((float)0.0,lower_price_range) || price > upper_price_range) {
        //std::cout << "bad price " << price << "\n";
        return false;
    }
    if (units <= 0){
        //std::cout << "bad units " << units << "\n";
        return false;
    }
    if (time.substr(0,8) != old_time.substr(0,8)){
        std::cout << time << "\n";
        return false;
    }
    if (std::abs(time_to_int(time)-time_to_int(old_time))>=2){
        std::cout << time << "\n";
        return false;
    }
    return true;
}
