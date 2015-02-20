#ifndef SCRUB_PROCESS_INCLUDED
#define SCRUB_PROCESS_INCLUDED

#include <sstream>
#include <list>
#include <algorithm>

int total_counter=0;
const int WINDOW_SIZE=100;

int get_line(char *mapped, std::stringstream &ss, int &location, int &end_location)
{
    int start_of_line = location;
    while (mapped[location] != '\n' && location < end_location){
        ss << mapped[location];
        ++location;
    }
    return start_of_line;
}

Tick find_median(std::list<Tick> &start_list)
{
    std::list<Tick>::iterator iter1 = start_list.begin();
    for (int i=0; i<WINDOW_SIZE/2; ++i) ++iter1;
    return *iter1;
}

void process_data(char *mapped, int start_location, int end_location, std::stringstream *noise)
{
    int location = start_location;
    std::stringstream ss;
    // find newline
    while (mapped[location] != '\n' && mapped[location] != '\0' && location != 0){
        ++location;
    }

    if (mapped[location] == '\n') ++location;

    if (location != start_location){
        Tick::bad_counter += 1; // for the skipped line in the beginning
        ++total_counter;
    }

    //figure out start data
    std::list<Tick> start_list;
    // loop through first WINDOW_SIZE lines and push into start_list.
    while (start_list.size() < WINDOW_SIZE && location < end_location){
        get_line(mapped, ss, location, end_location); // Grab the entire line
        if (location >= end_location) break;
        start_list.push_back(Tick(ss.str()));
        ss.str(std::string()); // Empty ss
        ++location;
    }

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


    std::list<Tick>::iterator iter = start_list.begin();
    for (int i=0; i<start_list.size(); ++i){

        if (iter->check_data(false)){
            *noise << iter->print() << "\n";
        }
        ++iter;
    }

    // Loop through entire file.

    int start_of_line;
    while (location < end_location){
        start_of_line = get_line(mapped, ss, location, end_location); // Grab the entire line
        if (location >= end_location) break;
        Tick obj(ss.str());
        if (obj.check_data()){
            *noise << ss.str() << "\n";
            mapped[start_of_line] = 'x';
        }
        ss.str(std::string()); // Empty ss
        ++location;
    }
    total_counter += Tick::counter;
}

#endif // SCRUB_PROCESS_INCLUDED