#ifndef SCRUB_FUNCTIONS_INCLUDED
#define SCRUB_FUNCTIONS_INCLUDED

#include <fstream>
#include <sstream>
#include <list>
#include <algorithm>
#include <sys/stat.h>

int total_counter=0; //total number of ticks
const int WINDOW_SIZE=100; //size of window used to get original comparison data

// Returns the number of characters in the file filename.
size_t getFilesize(const char* filename) {
    struct stat st;
    stat(filename, &st);
    return st.st_size;
}

// Copies the file. This is needed if you want the program to be run more than once.
// The scrub program does not rewrite all the good data. Instead it first copies the original file
// to a file called signal.txt and changes the bad ticks in that file so that the line starts with an 'x'
void copy_file(const char *from, const char *to)
{
    std::ifstream infile(from);
    std::ofstream outfile(to);

    outfile << infile.rdbuf();
    infile.close();
    outfile.close();
}

// Places a line from the input file into ss.
// Returns the integer location of the start of a newline.
int get_line(char *mapped, std::stringstream &ss, int &location, int &end_location)
{
    int start_of_line = location;
    while (mapped[location] != '\n' && location < end_location){
        ss << mapped[location];
        ++location;
    }
    return start_of_line;
}

// Finds the median of the list.
Tick find_median(std::list<Tick> &start_list)
{
    std::list<Tick>::iterator iter1 = start_list.begin();
    for (int i=0; i<WINDOW_SIZE/2; ++i) ++iter1;
    return *iter1;
}

// This is the meat of the program. It processes the data.
void process_data(char *mapped, int start_location, int end_location, std::stringstream *noise)
{
    int location = start_location;
    std::stringstream ss;
    // If this thread started in the middle of a line, we find the beginning of the next line.
    while (mapped[location] != '\n' && mapped[location] != '\0' && location != 0){
        ++location;
    }
    if (mapped[location] == '\n') ++location;

    int first_good_location = location; //first full line in thread starts here.

    // Since we skipped a line...
    if (location != start_location){
        Tick::bad_counter += 1;
        ++total_counter;
    }

    // When checking the data to see if its "bad", we need something to compare it to.
    // This finds the original data to compare it to.
    // It grabs WINDOW_SIZE data. It then sorts it by each characteristic (date,time,price,units)
    // and finds the median values. This is used as the "start data"
    std::list<Tick> start_list;
    // loop through first WINDOW_SIZE lines and push into start_list.
    while (start_list.size() < WINDOW_SIZE && location < end_location){
        get_line(mapped, ss, location, end_location); // Grab the entire line
        if (location >= end_location) break;
        start_list.push_back(Tick(ss.str(), true));
        ss.str(std::string()); // Empty ss
        ++location;
    }

    // All start data is in the list. Now find median for all 4 catagories.
    start_list.sort(compare_date);
    Tick::start_date = find_median(start_list).date;
    start_list.sort(compare_price);
    Tick::start_price = find_median(start_list).price;
    start_list.sort(compare_units);
    Tick::start_units = find_median(start_list).units;
    start_list.sort(compare_time);
    Tick::start_time = find_median(start_list).time;

    // Loop through entire file.
    location = first_good_location; //Go back to start of first full line in thread
    int start_of_line; //If data is bad, this will be changed to an 'x'
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

#endif // SCRUB_FUNCTIONS_INCLUDED
