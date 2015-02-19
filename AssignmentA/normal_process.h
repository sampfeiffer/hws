#ifndef PROCESS_INCLUDED
#define PROCESS_INCLUDED

#include <sstream>
#include <list>
#include <algorithm>

void get_line_normal(char *mapped, std::stringstream &ss, int &location, int &end_location)
{
    while (mapped[location] != '\n' && location < end_location){
        ss << mapped[location];
        ++location;
    }
}

void normal_process_data(char *mapped, int start_location, int end_location)
{
    int location = start_location;
    std::stringstream ss;
    // find newline
    while (mapped[location] != '\n' && mapped[location] != '\0' && location != 0){
        ++location;
    }

    if (mapped[location] == '\n') ++location;

    float old_price, new_price;
    bool is_first_line = true;


    // Loop through entire file. Add up the log return of the price.
    while (location < end_location){
        get_line_normal(mapped, ss, location, end_location); // Grab the entire line
        if (location >= end_location) break;
        ++location;
        if (is_first_line){
            is_first_line = false;
            continue;
        }
        Tick obj(ss.str());
        ss.str(std::string()); // Empty ss
    }
}

#endif // PROCESS_INCLUDED
