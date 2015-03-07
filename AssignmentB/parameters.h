#ifndef PARAMETERS_INCLUDED
#define PARAMETERS_INCLUDED

#include <iostream>
#include <fstream>

struct Parameters{

    float eur_usd_rate, cva_disc_rate;
    int counterparty_num, fx_num, swap_num, days_in_year, simulation_num;
    float time_horizon, step_size, recovery_rate, eur_usd_vol, amer_alphas[4], amer_sigmas[4], euro_alphas[4], euro_sigmas[4];
    float amer_betas[4], euro_betas[4];


    Parameters(const char* parameters_filename, const char* state0_filename); // Constructor
    const char* get_param(std::ifstream &infile);
    void print();
};

// Constructor. Read all the paramaters and initial states from the input files.
Parameters::Parameters(const char* parameters_filename, const char* state0_filename)
{
    std::ifstream parameters_infile, state0_infile;
    state0_infile.open(state0_filename);
    if (!state0_infile.is_open()){
        std::cout << "ERROR: state0.txt file could not be opened. Exiting.\n";
        exit(1);
    }

    eur_usd_rate = atof(get_param(state0_infile));
    state0_infile.close();

    parameters_infile.open(parameters_filename);
    if (!parameters_infile.is_open()){
        std::cout << "ERROR: parameters.txt file could not be opened. Exiting.\n";
        exit(1);
    }

    counterparty_num = atoi(get_param(parameters_infile));
    fx_num = atoi(get_param(parameters_infile));
    swap_num = atoi(get_param(parameters_infile));
    time_horizon = atof(get_param(parameters_infile));
    step_size = atof(get_param(parameters_infile));
    recovery_rate = atof(get_param(parameters_infile));
    eur_usd_vol = atof(get_param(parameters_infile));
    for (int i=0; i<4; ++i) amer_betas[i] = atof(get_param(parameters_infile));
    for (int i=0; i<4; ++i) amer_alphas[i] = atof(get_param(parameters_infile));
    for (int i=0; i<4; ++i) amer_sigmas[i] = atof(get_param(parameters_infile));
    for (int i=0; i<4; ++i) euro_betas[i] = atof(get_param(parameters_infile));
    for (int i=0; i<4; ++i) euro_alphas[i] = atof(get_param(parameters_infile));
    for (int i=0; i<4; ++i) euro_sigmas[i] = atof(get_param(parameters_infile));
    days_in_year = atoi(get_param(parameters_infile));
    cva_disc_rate = atof(get_param(parameters_infile));
    simulation_num = atoi(get_param(parameters_infile));

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
    std::cout << "\nInitial State of the world"
              << "\nEUR/USD rate: " << eur_usd_rate

              << "\n\nParameters"
              << "\nNum of counterparties: " << counterparty_num
              << "\nNum of FX: " << fx_num
              << "\nNum of swaps: " << swap_num
              << "\nTime horizon: " << time_horizon
              << "\nTime step size: " << step_size
              << "\nRecovery rate: " << recovery_rate
              << "\nEUR/USD volatility: " << eur_usd_vol;
    for (int i=0; i<4; ++i) std::cout << "\nAmer beta" << i << ": " << amer_betas[i];
    for (int i=0; i<4; ++i) std::cout << "\nAmer alpha" << i << ": " << amer_alphas[i];
    for (int i=0; i<4; ++i) std::cout << "\nAmer sigma" << i << ": " << amer_sigmas[i];
    for (int i=0; i<4; ++i) std::cout << "\nEuro beta" << i << ": " << euro_betas[i];
    for (int i=0; i<4; ++i) std::cout << "\nEuro alpha" << i << ": " << euro_alphas[i];
    for (int i=0; i<4; ++i) std::cout << "\nEuro sigma" << i << ": " << euro_sigmas[i];
    std::cout << "\nDays in a year: " << days_in_year
              << "\nCVA discount rate: " << cva_disc_rate
              << "\nNumber of Simulations: " << simulation_num << "\n\n";
}

#endif // PARAMETERS_INCLUDED
