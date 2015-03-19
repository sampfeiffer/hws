#ifndef PARAMETERS_INCLUDED
#define PARAMETERS_INCLUDED

#include <iostream>
#include <fstream>
#include <string>

struct Parameters{

    int time_bet_ticks, chars_per_line;
    float standard_error;

    Parameters(const char* parameters_filename); // Constructor
    const char* get_param(std::ifstream &infile);
    void print();
};

// Constructor. Read all the paramaters from the input files.
Parameters::Parameters(const char* parameters_filename)
{
    std::ifstream parameters_infile;
    parameters_infile.open(parameters_filename);
    if (!parameters_infile.is_open()){
        std::cout << "ERROR: parameters.txt file could not be opened. Exiting.\n";
        exit(1);
    }

    time_bet_ticks = atoi(get_param(parameters_infile));
    standard_error = atof(get_param(parameters_infile));
    chars_per_line = atoi(get_param(parameters_infile));

    parameters_infile.close();
}

// Get a single parameter
const char* Parameters::get_param(std::ifstream &infile)
{
    std::string text;
    getline(infile, text, ','); // data is always right after a comma
    getline(infile, text);
    return text.c_str();
}

// Print all the parameters and initial state
void Parameters::print()
{
    std::cout << "\nParameters"
              << "\nAverage time between ticks in milliseconds: " << time_bet_ticks
              << "\nStandard error of original data: " << standard_error
              << "\nCharacters per line of data: " << chars_per_line << "\n\n";
}

#endif // PARAMETERS_INCLUDED
