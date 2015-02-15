#ifndef PROCESS_INCLUDED
#define PROCESS_INCLUDED

#include <sstream>
#include <list>
#include <list>
#include <algorithm>

int total_counter=0;
const int WINDOW_SIZE=100;

void get_line(char *mapped, std::stringstream &ss, int &location, int &end_location)
{
    while (mapped[location] != '\n' && location < end_location){
        ss << mapped[location];
        ++location;
    }
}

Tick find_median(std::list<Tick> &start_list)
{
    std::list<Tick>::iterator it = start_list.begin();
    for (int i=0; i<WINDOW_SIZE/2; ++i) ++it;
    return *it;
}

void process_data(char *mapped, int start_location, int end_location)
{
    int location = start_location;
    std::stringstream ss;

    // find newline
    while (mapped[location] != '\n' && mapped[location] != '\0' && location != 0){
        ++location;
    }
    while (mapped[location] == '\n') ++location;
    int good_start_spot=location; //store the location of first good spot.

    if (location != start_location){
        Tick::bad_counter += 1; // for the skipped line in the beginning
        ++total_counter;
    }

    //figure out start data
    std::list<Tick> start_list;
    int start_data_size=0;
    std::cout << "here1\n";
    // loop through first WINDOW_SIZE lines and push into start_list.
    while (start_data_size < WINDOW_SIZE && location < end_location){
        get_line(mapped, ss, location, end_location); // Grab the entire line
        if (location >= end_location) break;
        std::cout << "here2 " << start_data_size << "\n";
        Tick obj(ss.str(),true);
        std::cout << "here3 " << start_data_size << "\n";
        start_list.push_back(obj);
        std::cout << "here4 " << start_data_size << "\n";
        ++start_data_size;
        ss.str(std::string()); // Empty ss
        ++location;
    }
    std::cout << "here5\n";

    // All start data is in list. Now find median for all 4 catagories.
    // Use the medians as the reference point data to check if other data is bad.
    start_list.sort(compare_date);
    Tick::start_date = find_median(start_list).date;
    start_list.sort(compare_price);
    Tick::start_price = find_median(start_list).price;
    start_list.sort(compare_units);
    Tick::start_units = find_median(start_list).units;
    start_list.sort(compare_time);
    Tick::start_time = find_median(start_list).time;

    location = good_start_spot; // Reset to first good line

//    for (int i=0; i<start_list.size(); ++i){
//        obj.check_data();
//    }

    // Loop through entire file.


    while (location < end_location){
        get_line(mapped, ss, location, end_location); // Grab the entire line
        if (location >= end_location) break;
        Tick obj(ss.str());
        obj.check_data();
        ss.str(std::string()); // Empty ss
        ++location;
    }
    total_counter += Tick::counter;
}

#endif // PROCESS_INCLUDED
