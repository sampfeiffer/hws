#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cstdlib>
#include <chrono>
#include <algorithm>
#include <thread>

// Only ignore data if it is very far off?
// What about zero size?
// How to test on own computer?
// Do we need to anticipate different date discrepencies?
// stuff you sent today?

//test for duplicate ticks
//ignore tiny volume
//parallel makefiles
// argv and argc


// Needed to time things.
#define TIMER_OBJ std::chrono::steady_clock::time_point
#define NOW std::chrono::steady_clock::now()
#define ELAPSED_TIME std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count()

const int TEST_SIZE=1000, PRICE_MULTIPLE=4;
static const int NUM_THREADS = 10;

struct Data{
    std::string time;
    float price;
    int units;

    Data(std::string time_, float _price, int _units){
        time = time_;
        price = _price;
        units = _units;
    }
};

// Converts the time part of the date string to an int. Ignores fractions of a second.
int time_to_int(std::string time)
{
    float num=0;
    num+=atoi(time.substr(9,2).c_str())*3600;
    num+=atoi(time.substr(12,2).c_str())*60;
    num+=atoi(time.substr(15,2).c_str());
    return num;
}

bool compare_time(const Data &obj1, const Data &obj2)
{
    return (obj1.time < obj2.time);
}

bool compare_price(const Data &obj1, const Data &obj2)
{
    return (obj1.price < obj2.price);
}

// Checks if data is good.
bool is_data_good(std::string time, float price, int units, float lower_price_range, float upper_price_range, std::string old_time)
{
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

//test
void ThreadFunction(int threadID) {
    std::cout << "Hello from thread #" << threadID << std::endl;
}

int main(int argc, char *argv[])
{

    //open file
    // read beginning data. and find medians.
    // find valid starting date
    // find valid volume
    // find valid price
    //read each line. put into data structure. check if data is good.


    TIMER_OBJ t0 = NOW;
    std::string filename = "data10k.txt";
    std::string time_, price_str_, units_str_, old_time="";
    float _price;
    int _units;
    std::vector<Data> lines;
    std::ifstream infile;
    bool is_data_good(std::string time, float price, int units, float lower_price_range, float upper_price_range, std::string old_time);

    infile.open(filename);
    if (!infile.is_open()){
        std::cout << "error\n";
    }

    // Figure out upper and lower bound.
    for (int i=0; i<TEST_SIZE; ++i){
        if (!getline(infile, time_, ',')) break;
        getline(infile, price_str_, ',');
        _price = atof(price_str_.c_str());
        getline(infile, units_str_);
        _units = atoi(units_str_.c_str());
        lines.push_back(Data(time_, _price, _units));
    }
    std::sort(lines.begin(), lines.end(), compare_price);
    float q2, inter_quartile, lower_price_range, upper_price_range;
    q2 = lines[lines.size()/2].price;
    inter_quartile = lines[lines.size()*3/4].price - lines[lines.size()/4].price;
    lower_price_range = q2-inter_quartile*PRICE_MULTIPLE;
    upper_price_range = q2+inter_quartile*PRICE_MULTIPLE;
    //std::cout << "lower_price_range " << lower_price_range << "\n";
    //std::cout << "upper_price_range " << upper_price_range << "\n";

    lines.clear();
    infile.seekg(0, infile.beg);

    // Read data and remove bad entries.
    while (!infile.eof()){
        if (!getline(infile, time_, ',')) break;
        //time_int = time_to_int(time_);
        if (old_time =="") old_time = time_;
        getline(infile, price_str_, ',');
        _price = atof(price_str_.c_str());
        getline(infile, units_str_);
        _units = atoi(units_str_.c_str());
        if (is_data_good(time_, _price, _units, lower_price_range, upper_price_range, old_time)){
            lines.push_back(Data(time_, _price, _units));
            old_time = time_;
        }
        //cout << time_ << " " << _price << " " << _units << "\n";
    }

    std::thread thread[NUM_THREADS];
    //std::cout << std::thread::hardware_concurrency() << "\n";
    // Launch threads.
    for (int i = 0; i < NUM_THREADS; ++i) {
        thread[i] = std::thread(ThreadFunction, i);
    }
    std::cout << NUM_THREADS << " threads launched." << std::endl;
    // Join threads to the main thread of execution.
    for (int i = 0; i < NUM_THREADS; ++i) {
        thread[i].join();
    }

    infile.close();
    TIMER_OBJ t1 = NOW;

//    for (unsigned int i=0; i<lines.size(); ++i){
//        std::cout << lines[i].time << " " << lines[i].price << " " << lines[i].units << "\n";
//    }

    std::cout << "\nsize: " << lines.size() << "\n";
    std::cout << "Read data - time elapsed: " << ELAPSED_TIME << "ms\n";

    t0 = NOW;
    std::sort(lines.begin(), lines.end(), compare_time);
    t1 = NOW;

    std::cout << "Sort data - time elapsed: " << ELAPSED_TIME << "ms\n\n";

    return 0;
}
