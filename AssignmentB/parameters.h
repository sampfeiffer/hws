#ifndef PARAMETERS_INCLUDED
#define PARAMETERS_INCLUDED

#include <iostream>
#include <fstream>

struct Parameters{

    std::ifstream parameters_infile;
    int counterparty_num, fx_num, swap_num;
    float time_horizon, step_size, recovery_rate, eur_usd_vol, ns_vol, mean_revert;
    double beta0, beta1, beta2, tau;


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
    time_horizon = atof(get_param());
    step_size = atof(get_param());
    recovery_rate = atof(get_param());
    eur_usd_vol = atof(get_param());
    beta0 = atof(get_param());
    beta1 = atof(get_param());
    beta2 = atof(get_param());
    tau = atof(get_param());
    mean_revert = atof(get_param());
    ns_vol = atof(get_param());

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
    std::cout << "\nParameters"
              << "\nNum of counterparties: " << counterparty_num
              << "\nNum of FX: " << fx_num
              << "\nNum of swaps: " << swap_num
              << "\nTime horizon: " << time_horizon
              << "\nTime step size: " << step_size
              << "\nRecovery rate: " << recovery_rate
              << "\nEUR/USD volatility: " << eur_usd_vol
              << "\nbeta0: " << beta0
              << "\nbeta1: " << beta1
              << "\nbeta2: " << beta2
              << "\ntau: " << tau
              << "\nMean reversion rate: " << mean_revert
              << "\nNelson-Siegel volatility: " << ns_vol << "\n";
}

#endif // PARAMETERS_INCLUDED
