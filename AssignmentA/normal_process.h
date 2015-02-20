#ifndef NORMAL_PROCESS_INCLUDED
#define NORMAL_PROCESS_INCLUDED

#include <sstream>
#include <cmath>

void get_line_normal(char *mapped, std::stringstream &ss, int &location, int &end_location)
{
    while (mapped[location] != '\n' && location < end_location){
        ss << mapped[location];
        ++location;
    }
}

void normal_process_data(char *mapped, int start_location, int end_location, int *jb)
{
    int location = start_location;
    std::stringstream ss;
    // find newline
    while (mapped[location] != '\n' && mapped[location] != '\0' && location != 0){
        ++location;
    }

    if (mapped[location] == '\n') ++location;

    int good_start_location = location;

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

    double mean_return = sum_of_returns/counter;
    double sum_of_error2 = 0, sum_of_error3 = 0, sum_of_error4 = 0;
    double diff;

    location = good_start_location;
    int temp_counter=1;
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
        ++temp_counter;
    }

    double skewness = (sum_of_error3/counter) / pow(sum_of_error2/counter, 1.5);
    double kurtosis = (sum_of_error4/counter) / pow(sum_of_error2/counter, 2);

    *jb = (counter/6.0) * (pow(skewness,2) + 0.25*pow(kurtosis-3,2));

//    std::cout << "counter: " << counter << "\n";
//    std::cout << "mean_return: " << mean_return << "\n";
//    std::cout << "skewness: " << skewness << "\n";
//    std::cout << "kurtosis: " << kurtosis << "\n";
//    std::cout << "jarque_bera: " << jb << "\n\n";

}

#endif // NORMAL_PROCESS_INCLUDED
