#ifndef PROCESS_INCLUDED
#define PROCESS_INCLUDED

#include <sstream>
#include <list>
#include <vector>
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

void process_data(char *mapped, int start_location, int end_location)
{
    int location = start_location;
    std::stringstream ss;

    std::cout << "here1\n";

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
    std::cout << "here2\n";

    //figure out start data
    std::vector<Tick> start_list;
    int start_data_size=0;
    // loop through first WINDOW_SIZE lines and push into start_list.
    while (start_data_size < WINDOW_SIZE && location < end_location){
        get_line(mapped, ss, location, end_location); // Grab the entire line
        if (location >= end_location) break;
        start_list.push_back(Tick(ss.str(), true));
        ++start_data_size;
        ss.str(std::string()); // Empty ss
        ++location;
    }
    std::cout << "here3\n";

    // All start data is in vector. Now find median for all 4 catagories.
    // Use the medians as the reference point data to check if other data is bad.
    std::sort(start_list.begin(), start_list.end(), compare_date);
    Tick::start_date = start_list[start_list.size()/2].date;
    std::sort(start_list.begin(), start_list.end(), compare_price);
    Tick::start_price = start_list[start_list.size()/2].price;
    std::sort(start_list.begin(), start_list.end(), compare_units);
    Tick::start_units = start_list[start_list.size()/2].units;
    std::sort(start_list.begin(), start_list.end(), compare_time);
    Tick::start_time = start_list[start_list.size()/2].time;

    location = good_start_spot; // Reset to first good line

//    for (int i=0; i<start_list.size(); ++i){
//        obj.check_data();
//    }
    std::cout << "here4\n";
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
    std::cout << "here5\n";
}

#endif // PROCESS_INCLUDED