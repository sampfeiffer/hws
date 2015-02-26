#include <random>
#include "parameters.h"
#include "state0.h"

float nelson_siegel(float t, double beta0, float beta1, float beta2, float tau)
{
    return beta0 + beta1*(1-exp(-t/tau))/(t/tau) + beta2*((1-exp(-t/tau))/(t/tau)-exp(-t/tau));
}

int main(int argc, char *argv[])
{
    std::string parameters_filename="parameters.txt", state0_filename="state0.txt";
    std::default_random_engine generator;

    // Get parameters and initial state of the world.
    Parameters params(parameters_filename);
    params.print();
    State0 initial_state(state0_filename);
    initial_state.print();

    // Generate the path of the fx rate and Nelson-Siegel parameters
    std::normal_distribution<double> normal_fx(0.0,params.eur_usd_vol*sqrt(params.step_size/365));
    std::normal_distribution<double> normal_ns(0.0,params.ns_vol*sqrt(params.step_size/365));
    int num_of_steps = 365*params.time_horizon/params.step_size;
    double current_fx_rate = initial_state.eur_usd_rate;
    double current_beta0 = params.beta0;
    double current_beta1 = params.beta1;
    double current_beta2 = params.beta2;
    double current_tau = params.tau;

    for (int i=1; i<num_of_steps; ++i){
        //generate current state of the world
        current_fx_rate += normal_fx(generator);
        current_beta0 = params.beta0 + params.mean_revert*(params.beta0-current_beta0) + normal_ns(generator);
        current_beta1 = params.beta1 + params.mean_revert*(params.beta1-current_beta1) + normal_ns(generator);
        current_beta2 = params.beta2 + params.mean_revert*(params.beta2-current_beta2) + normal_ns(generator);
        current_tau = params.tau + params.mean_revert*(params.tau-current_tau) + normal_ns(generator);

        /*std::cout << nelson_siegel(1,current_beta0,current_beta1,current_beta2,current_tau) << " "
                  << nelson_siegel(2,current_beta0,current_beta1,current_beta2,current_tau) << " "
                  << nelson_siegel(5,current_beta0,current_beta1,current_beta2,current_tau) << " "
                  << nelson_siegel(10,current_beta0,current_beta1,current_beta2,current_tau) << " "
                  << nelson_siegel(20,current_beta0,current_beta1,current_beta2,current_tau) << "\n";*/


    }


    std::cout << "\n";

    return 0;
}
