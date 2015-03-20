#ifndef PARAMETERS_INCLUDED
#define PARAMETERS_INCLUDED

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>

struct Parameters{

    int time_bet_ticks, chars_per_line;
    float standard_error;
    std::string input_data_filename;
    char* tick_data_filename;

    Parameters(const char* parameters_filename); // Constructor
    ~Parameters(); // Destructor
    const char* get_param(std::ifstream &infile);
    std::string get_param_string(std::ifstream &infile);
    std::string print();
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
    input_data_filename = get_param_string(parameters_infile);
    std::string temp_filename = get_param_string(parameters_infile);
    tick_data_filename = new char[temp_filename.size() + 1];
    std::copy(temp_filename.begin(), temp_filename.end(), tick_data_filename);
    tick_data_filename[temp_filename.size()] = '\0';

    parameters_infile.close();
}

// Destructor
Parameters::~Parameters()
{
    delete [] tick_data_filename;
}

// Get a single parameter
const char* Parameters::get_param(std::ifstream &infile)
{
    std::string text;
    getline(infile, text, ','); // data is always right after a comma
    getline(infile, text);
    return text.c_str();
}

// Get a single parameter that is a string
std::string Parameters::get_param_string(std::ifstream &infile)
{
    std::string text;
    getline(infile, text, ','); // data is always right after a comma
    getline(infile, text);
    return text;
}

// Print all the parameters and initial state
std::string Parameters::print()
{
    std::stringstream ss;
    ss << "\nParameters"
              << "\nAverage time between ticks in milliseconds: " << time_bet_ticks
              << "\nStandard error of original data: " << standard_error
              << "\nCharacters per line of data: " << chars_per_line
              << "\nOriginal data filename: " << input_data_filename
              << "\nTick data filename: " << tick_data_filename << "\n\n";
    return ss.str();
}

#endif // PARAMETERS_INCLUDED
