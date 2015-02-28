#ifndef PARAMETERS_INCLUDED
#define PARAMETERS_INCLUDED

#include <iostream>
#include <fstream>

struct Parameters{

    std::ifstream parameters_infile;
    int counterparty_num, fx_num, swap_num, deals_at_once;
    float time_horizon, step_size, recovery_rate, eur_usd_vol, amer_alphas[4], amer_sigmas[4], euro_alphas[4], euro_sigmas[4];
    double amer_betas[4], euro_betas[4];


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
    for (int i=0; i<4; ++i) amer_betas[i] = atof(get_param());
    for (int i=0; i<4; ++i) amer_alphas[i] = atof(get_param());
    for (int i=0; i<4; ++i) amer_sigmas[i] = atof(get_param());
    for (int i=0; i<4; ++i) euro_betas[i] = atof(get_param());
    for (int i=0; i<4; ++i) euro_alphas[i] = atof(get_param());
    for (int i=0; i<4; ++i) euro_sigmas[i] = atof(get_param());
    deals_at_once = atoi(get_param());

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
              << "\nEUR/USD volatility: " << eur_usd_vol;
    for (int i=0; i<4; ++i) std::cout << "\nAmer beta" << i << ": " << amer_betas[i];
    for (int i=0; i<4; ++i) std::cout << "\nAmer alpha" << i << ": " << amer_alphas[i];
    for (int i=0; i<4; ++i) std::cout << "\nAmer sigma" << i << ": " << amer_sigmas[i];
    for (int i=0; i<4; ++i) std::cout << "\nEuro beta" << i << ": " << euro_betas[i];
    for (int i=0; i<4; ++i) std::cout << "\nEuro alpha" << i << ": " << euro_alphas[i];
    for (int i=0; i<4; ++i) std::cout << "\nEuro sigma" << i << ": " << euro_sigmas[i];
    std::cout << "\nDeals handled at once: " << deals_at_once << "\n";
}

#endif // PARAMETERS_INCLUDED
