#ifndef PROCESS_INCLUDED
#define PROCESS_INCLUDED

#include <sstream>
#include <vector>
#include <algorithm>

int total_counter=0;

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
    std::vector<Tick> start_vector;
    int start_data_size=0;
    // loop through first 100 lines and push into start_vector.
    while (start_data_size < 100 && location < end_location){
        // Grab the entire line
        while (mapped[location] != '\n' && location < end_location){
            ss << mapped[location];
            ++location;
        }
        if (location >= end_location) break;
        std::string data_string = ss.str();
        start_vector.push_back(Tick(data_string, true));
        ++start_data_size;
        ss.str(std::string()); // Empty ss
        ++location;
    }

    // All start data is in vector. Now find median for all 4 catagories.
    // Use the medians as the reference point data to check if other data is bad.
    std::sort(start_vector.begin(), start_vector.end(), compare_date);
    Tick::start_date = start_vector[start_vector.size()/2].date;
    std::sort(start_vector.begin(), start_vector.end(), compare_time);
    Tick::start_time = start_vector[start_vector.size()/2].time;
    std::sort(start_vector.begin(), start_vector.end(), compare_price);
    Tick::start_price = start_vector[start_vector.size()/2].price;
    std::sort(start_vector.begin(), start_vector.end(), compare_units);
    Tick::start_units = start_vector[start_vector.size()/2].units;

    location = good_start_spot; // Reset to first good line

    // Loop through entire file.
    while (location < end_location){
        // Grab the entire line
        while (mapped[location] != '\n' && location < end_location){
            ss << mapped[location];
            ++location;
        }
        if (location >= end_location) break;
        std::string data_string = ss.str();
        Tick obj(data_string);
        obj.check_data();
        ss.str(std::string()); // Empty ss
        ++location;
    }
    total_counter += Tick::counter;
}

#endif // PROCESS_INCLUDED
