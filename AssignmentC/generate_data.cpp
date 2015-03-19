#include <random>
#include <cmath>
#include <time.h>
#include "parameters.h"

//change average time between ticks in milliseconds to 87 to make 45GB

// Gets the next piece of data from the input file
void get_next_data(std::ifstream &input_data_infile, double &data)
{
    input_data_infile >> data; //Skip the date
    input_data_infile >> data;
}

int main(int argc, char *argv[])
{
    void get_next_data(std::ifstream &input_data_infile, double &data);
    const char* parameters_filename = "parameters.txt";
    const char* input_data_filename = "input_data.txt";
    const char* tick_data_filename = "tick_data.dat";
    std::ifstream input_data_infile;
    std::ofstream tick_data_outfile;

    int milliseconds_in_day = 24*60*60*1000;
    double fed_rate_old, fed_rate_new, tick_value, drift_per_tick;

    // Setup the standard normal generator
    srand(time(NULL));
    std::default_random_engine generator;
    std::normal_distribution<double> standard_normal(0,1);

    // Get (and print) the program paramters.
    Parameters params(parameters_filename);
    params.print();

    // Determine the number of ticks per day and the standard deviation of each tick.
    int ticks_per_day = milliseconds_in_day/params.time_bet_ticks - 1;
    double stdev_per_tick = params.standard_error/std::sqrt(ticks_per_day);

    // Open the input and output files
    input_data_infile.open(input_data_filename);
    if (!input_data_infile.is_open()){
        std::cout << "ERROR: input_data.txt file could not be opened. Exiting.\n";
        exit(1);
    }
    tick_data_outfile.open(tick_data_filename);
    if (!tick_data_outfile.is_open()){
        std::cout << "ERROR: tick_data.dat file could not be opened. Exiting.\n";
        exit(1);
    }
    tick_data_outfile.setf(std::ios::fixed, std:: ios::floatfield);

    // Get the initial fed fund rate
    get_next_data(input_data_infile, fed_rate_old);
    tick_data_outfile << fed_rate_old << "\n";

    // For each daily fed fund rate, create the appropriate number of intraday ticks
    while (!input_data_infile.eof()){
        get_next_data(input_data_infile, fed_rate_new);
        drift_per_tick = (fed_rate_new - fed_rate_old)/ticks_per_day;

        // Fill in the tick data
        tick_value = fed_rate_old;
        for (int j=0; j<ticks_per_day; ++j){
            tick_value += drift_per_tick + stdev_per_tick*standard_normal(generator);
            tick_data_outfile << tick_value << "\n";
        }

        tick_data_outfile << fed_rate_new << "\n";
        fed_rate_old = fed_rate_new;
    }

    input_data_infile.close();
    tick_data_outfile.close();
    std::cout << "\n";
    return 0;
}
