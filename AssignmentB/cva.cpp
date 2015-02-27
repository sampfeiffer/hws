#include <random>
#include <vector>
#include "parameters.h"
#include "state0.h"
#include "counterparty.h"

float nelson_siegel(float t, double beta0, float beta1, float beta2, float tau)
{
    return beta0 + beta1*(1-exp(-t/tau))/(t/tau) + beta2*((1-exp(-t/tau))/(t/tau)-exp(-t/tau));
}

int main(int argc, char *argv[])
{
    std::string parameters_filename="parameters.txt", state0_filename="state0.txt", sizes_filename="sizes.txt",
                counterparty_deals_filename="counterparty_deals.txt",
                fx_details_filename="fx_details.txt", swap_details_filename="swap_details.txt";
    std::default_random_engine generator;
    std::ifstream counterparty_deals_infile, fx_details_infile, swap_details_infile, sizes_infile;

    // Get parameters and initial state of the world.
    Parameters params(parameters_filename);
    params.print();
    State0 initial_state(state0_filename);
    initial_state.print();

    int sizes[5];
    sizes_infile.open(sizes_filename);
    if (!sizes_infile.is_open()){
        std::cout << "ERROR: sizes.txt file could not be opened. Exiting.\n";
        exit(1);
    }
    for (int i=0; i<5; ++i){
        sizes_infile >> sizes[i];
    }
    sizes_infile.close();

    // Generate the path of the fx rate and Nelson-Siegel parameters
    std::normal_distribution<double> normal_fx(0.0,params.eur_usd_vol*sqrt(params.step_size/365));
    std::normal_distribution<double> normal_ns(0.0,params.ns_vol*sqrt(params.step_size/365));
    int num_of_steps = 365*params.time_horizon/params.step_size;
    double current_fx_rate = initial_state.eur_usd_rate;
    double current_beta0 = params.beta0;
    double current_beta1 = params.beta1;
    double current_beta2 = params.beta2;
    double current_tau = params.tau;

    counterparty_deals_infile.open(counterparty_deals_filename);
    if (!counterparty_deals_infile.is_open()){
        std::cout << "ERROR: counterparty_deals.txt file could not be opened. Exiting.\n";
        exit(1);
    }
    fx_details_infile.open(fx_details_filename);
    if (!fx_details_infile.is_open()){
        std::cout << "ERROR: fx_details.txt file could not be opened. Exiting.\n";
        exit(1);
    }
    swap_details_infile.open(swap_details_filename);
    if (!swap_details_infile.is_open()){
        std::cout << "ERROR: swap_details.txt file could not be opened. Exiting.\n";
        exit(1);
    }

    int current_id=1, deal_id, id=1, deals_handled=0;
    std::vector<Counterparty> cp_vector;
    std::string deal_text;
    counterparty_deals_infile >> deal_id;

    while (deals_handled <= params.deals_at_once){
        Counterparty cp(id);
        do{
            counterparty_deals_infile >> deal_id;
            if (deal_id<params.fx_num){
                getline(fx_details_infile, deal_text);
                cp.add_fx(deal_text);
            }
            else {
                getline(swap_details_infile, deal_text);
                cp.add_swap(deal_text);
            }
            ++deals_handled;
            counterparty_deals_infile >> current_id;
        } while(current_id == id);
        cp_vector.push_back(cp);
        ++id;
    }

//    std::cout << "size " << cp_vector.size() << "\n";
//    for (unsigned int i=0; i<cp_vector.size(); ++i){
//        cp_vector[i].print();
//    }
//    std::cout << "steps " << num_of_steps << "\n";
    //for (int i=1; i<=num_of_steps; ++i){
    for (int i=1; i<=1; ++i){
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


        //cp_vector[0].fx_deals[0].print();
        std::cout << "value " << cp_vector[0].fx_deals[0].value(initial_state.eur_usd_rate, current_fx_rate) << "\n";

    }
    counterparty_deals_infile.close();
    fx_details_infile.close();
    swap_details_infile.close();

    std::cout << "\n";

    return 0;
}
