#include <random>
#include <vector>
#include "parameters.h"
#include "state0.h"
#include "counterparty.h"
#include "nelson_siegel.h"

float nelson_siegel(float t, double beta0, float beta1, float beta2, float tau)
{
    return beta0 + beta1*(1-exp(-t/tau))/(t/tau) + beta2*((1-exp(-t/tau))/(t/tau)-exp(-t/tau));
}

int main(int argc, char *argv[])
{
    std::string parameters_filename="parameters.txt", state0_filename="state0.txt", hazard_buckets_filename="hazard_buckets.txt",
                counterparty_deals_filename="counterparty_deals.txt",
                fx_details_filename="fx_details.txt", swap_details_filename="swap_details.txt";
    std::default_random_engine generator;
    std::ifstream counterparty_deals_infile, fx_details_infile, swap_details_infile, hazard_buckets_infile;

    // Get parameters and initial state of the world.
    Parameters params(parameters_filename);
    params.print();
    State0 initial_state(state0_filename);
    initial_state.print();

    int hazard_buckets[5];
    hazard_buckets_infile.open(hazard_buckets_filename);
    if (!hazard_buckets_infile.is_open()){
        std::cout << "ERROR: hazard_buckets.txt file could not be opened. Exiting.\n";
        exit(1);
    }
    for (int i=0; i<5; ++i){
        hazard_buckets_infile >> hazard_buckets[i];
    }
    hazard_buckets_infile.close();

    // Generate the path of the fx rate and Nelson-Siegel parameters
    std::normal_distribution<double> normal_fx(0.0,params.eur_usd_vol*sqrt(params.step_size/365));
    int num_of_steps = 365*params.time_horizon/params.step_size;
    double current_fx_rate = initial_state.eur_usd_rate;
    NelsonSiegel amer(params.step_size, params.amer_betas, params.amer_alphas, params.amer_sigmas);

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

    int current_id=1, deal_id, id=1, deals_handled=0, bucket=0;
    float hazard_rate=0.10;
    std::vector<Counterparty> cp_vector;
    std::string deal_text;
    counterparty_deals_infile >> deal_id;

    while (deals_handled <= params.deals_at_once){
        if (id > hazard_buckets[bucket]){
            ++bucket;
            hazard_rate -= 0.02;
        }
        Counterparty cp(id, hazard_rate);
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

    //for (int i=1; i<=num_of_steps; ++i){
    for (int i=1; i<=30; ++i){
        //generate current state of the world
        current_fx_rate += normal_fx(generator);
        amer.sim_next_step();


        //cp_vector[0].fx_deals[0].print();
        std::cout << "value " << cp_vector[0].fx_deals[0].value(initial_state.eur_usd_rate, current_fx_rate) << "\n";

    }
    counterparty_deals_infile.close();
    fx_details_infile.close();
    swap_details_infile.close();

    std::cout << "\n";

//    NelsonSiegel test(params.step_size, params.amer_betas, params.amer_alphas, params.amer_sigmas);
//    std::cout << "yield " << test.yield(4) << "\n";
//    for (int i=0; i<200; ++i){
//        test.sim_next_step();
//        std::cout << "yield " << test.yield(4) << "\n";
//    }

    return 0;
}
