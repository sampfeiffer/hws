#ifndef PARAMETERS_INCLUDED
#define PARAMETERS_INCLUDED

#include <iostream>
#include <fstream>

struct Parameters{

    std::ifstream parameters_infile;
    int counterparty_num;
    int fx_num;
    int swap_num;

    Parameters(std::string parameters_filename);
    const char* get_param();
    void print();
};

Parameters::Parameters(std::string parameters_filename)
{
    parameters_infile.open(parameters_filename);
    if (!parameters_infile.is_open()){
        std::cout << "ERROR: parameters.txt file could not be opened. Exiting.\n";
        exit(1);
    }

    counterparty_num = atoi(get_param());
    fx_num = atoi(get_param());
    swap_num = atoi(get_param());

    parameters_infile.close();
}

const char* Parameters::get_param()
{
    std::string text;
    getline(parameters_infile, text, ',');
    getline(parameters_infile, text);
    return text.c_str();
}

void Parameters::print()
{
    std::cout << "Num of counterparties: " << counterparty_num
              << "\nNum of FX: " << fx_num
              << "\nNum of swaps: " << swap_num << "\n";
}

#endif // PARAMETERS_INCLUDED
