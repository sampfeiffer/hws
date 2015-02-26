#ifndef STATE0_INCLUDED
#define STATE0_INCLUDED

#include <iostream>
#include <fstream>

struct State0{

    std::ifstream state0_infile;
    float eur_usd_rate;

    State0(std::string state0_filename);
    const char* get_param();
    void print();
};

State0::State0(std::string state0_filename)
{
    state0_infile.open(state0_filename);
    if (!state0_infile.is_open()){
        std::cout << "ERROR: state0.txt file could not be opened. Exiting.\n";
        exit(1);
    }

    eur_usd_rate = atof(get_param());

    state0_infile.close();
}

const char* State0::get_param()
{
    std::string text;
    getline(state0_infile, text, ',');
    getline(state0_infile, text);
    return text.c_str();
}

void State0::print()
{
    std::cout << "\nInitial State of the world"
              << "\nEUR/USD rate: " << eur_usd_rate << "\n";
}

#endif // STATE0_INCLUDED
