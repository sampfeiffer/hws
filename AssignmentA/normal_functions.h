#ifndef NORMAL_FUNCTIONS_INCLUDED
#define NORMAL_FUNCTIONS_INCLUDED

#include <sstream>
#include <cmath>
#include <sys/stat.h>
#include "normal_tick.h"

// Returns the number of characters in the file filename.
size_t getFilesize(const char* filename) {
    struct stat st;
    stat(filename, &st);
    return st.st_size;
}

// Places a line from the input file into ss.
void get_line_normal(char *mapped, std::stringstream &ss, int &location, int &end_location)
{
    while (mapped[location] != '\n' && location < end_location){
        ss << mapped[location];
        ++location;
    }
}

// This is the meat of the program. It processes the data and finds the Jarque-Bera number
// for the normality testing.
void normal_process_data(char *mapped, int start_location, int end_location, long *jb)
{
    int location = start_location;
    std::stringstream ss;
    // find newline
    while (mapped[location] != '\n' && mapped[location] != '\0' && location != 0){
        ++location;
    }
    if (mapped[location] == '\n') ++location;

    int first_good_location = location; //first full line in thread starts here.

    float old_price;
    bool is_first_line = true;
    double sum_of_returns = 0;
    int counter=0;

    // Loop through entire file. Add up the log return of the price.
    while (location < end_location){
        ss.str(std::string()); // Empty ss
        get_line_normal(mapped, ss, location, end_location); // Grab the entire line
        if (location >= end_location) break;
        ++location;
        if (ss.str()[0] == 'x'){
            continue;
        }
        Tick obj(ss.str());
        if (is_first_line){
            is_first_line = false;
            old_price = obj.price;
            continue;
        }

        sum_of_returns += log(obj.price/old_price);
        old_price = obj.price;
        ++counter;

    }

    double mean_return = sum_of_returns/counter; //calculate the mean of log returns
    double diff, sum_of_error2 = 0, sum_of_error3 = 0, sum_of_error4 = 0;

    location = first_good_location;
    is_first_line = true;

    // Loop through entire file. Add up the squared, cubed, and fourth, difference.
    while (location < end_location){
        ss.str(std::string()); // Empty ss
        get_line_normal(mapped, ss, location, end_location); // Grab the entire line
        if (location >= end_location) break;
        ++location;
        if (ss.str()[0] == 'x') continue; // If the data is flagged as bad, skip that line.
        Tick obj(ss.str());
        if (is_first_line){
            is_first_line = false;
            old_price = obj.price;
            continue;
        }

        diff = log(obj.price/old_price)-mean_return;
        sum_of_error2 += pow(diff, 2);
        sum_of_error3 += pow(diff, 3);
        sum_of_error4 += pow(diff, 4);

        old_price = obj.price;
    }

    double skewness = (sum_of_error3/counter) / pow(sum_of_error2/counter, 1.5);
    double kurtosis = (sum_of_error4/counter) / pow(sum_of_error2/counter, 2);

    *jb = (counter/6.0) * (pow(skewness,2) + 0.25*pow(kurtosis-3,2)); //Jarque_Bera number
}

#endif // NORMAL_FUNCTIONS_INCLUDED
