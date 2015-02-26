#include <random>
#include "parameters.h"
#include "state0.h"

int main(int argc, char *argv[])
{
    std::string parameters_filename="parameters.txt", state0_filename="state0.txt";
    std::default_random_engine generator;

    // Get parameters and initial state of the world.
    Parameters params(parameters_filename);
    params.print();
    State0 initial_state(state0_filename);
    initial_state.print();

    // Generate the path of the fx rate
    std::normal_distribution<double> normal_dist(0.0,params.eur_usd_vol*sqrt(params.step_size));
    int num_of_steps = params.time_horizon/params.step_size;
    double current_fx_rate = initial_state.eur_usd_rate;
    std::cout << current_fx_rate << "\n";
    for (int i=1; i<num_of_steps; ++i){
        current_fx_rate += normal_dist(generator);
        std::cout << current_fx_rate << "\n";
    }


    std::cout << "\n";
    return 0;
}
