#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/random/linear_congruential_engine.h>
#include <thrust/random/normal_distribution.h>

#include <vector>
#include <algorithm>
#include "parameters.h"
#include "counterparty.h"
#include "state.h"

struct calculate_cva{
    Parameters params;

    calculate_cva(Parameters &params_) : params(params_)
    {}
    __host__
    float operator()(Counterparty &cp) {
        float temp1;
        float temp2;
        float cva=0;
        int num_of_steps = 360*params.time_horizon/params.step_size;
        State world_state(params);
        for (int i=0; i<num_of_steps; ++i){
            world_state.sim_next_step();
            // CVA for fx
            for (unsigned int fx=0; fx<cp.num_of_fx; ++fx){
                temp1 = world_state.cva_disc_factor;
                temp2 = cp.prob_default(world_state.time);
                cva += temp1*temp2* std::max(cp.fx_deals[fx]->value(world_state.fx_rate_beg, world_state.fx_rate),0.0);
            }
            // CVA for swaps
            for (unsigned int sw=0; sw<cp.num_of_swap; ++sw){
                cva += world_state.cva_disc_factor * cp.prob_default(world_state.time)
                       * std::max(cp.swap_deals[sw]->value(world_state),0.0);
            }
        }
        cva *= 1-params.recovery_rate;
        return cva;
    }
};

int main(int argc, char *argv[])
{
    const char* parameters_filename="parameters.txt";
    const char* state0_filename="state0.txt";
    const char* hazard_buckets_filename="hazard_buckets.txt";
    const char* counterparty_deals_filename="counterparty_deals.txt";
    const char* fx_details_filename="fx_details.txt";
    const char* swap_details_filename="swap_details.txt";
    std::ifstream counterparty_deals_infile, fx_details_infile, swap_details_infile, hazard_buckets_infile;

    // Get parameters and initial state of the world.
    Parameters params(parameters_filename, state0_filename);
    params.print();

    // Get the list of hazard rate bucket endpoints
    int hazard_buckets[5];
    hazard_buckets_infile.open(hazard_buckets_filename);
    if (!hazard_buckets_infile.is_open()){
        std::cout << "ERROR: hazard_buckets.txt file could not be opened. Exiting.\n";
        exit(1);
    }
    for (int i=0; i<5; ++i) hazard_buckets_infile >> hazard_buckets[i];
    hazard_buckets_infile.close();


    // Open the counterparty deals and deal details
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

    // Read deals into memory
    int current_id=1, deal_id, id=1, deals_handled=0, bucket=0;
    float hazard_rate=0.10;

    //thrust::device_vector<Counterparty> cp_vector;
    //thrust::device_vector<float> cva_vector;
    thrust::host_vector<Counterparty> cp_vector;
    thrust::host_vector<float> cva_vector;
    //std::vector<Counterparty> cp_vector;
    //std::vector<float> cva_vector;

    int fx_id, swap_id, notional, tenor, start_of_data, fx_count, swap_count;
    char position, denomination;
    float fixed_rate;
    counterparty_deals_infile >> deal_id;

    while (deals_handled <= params.deals_at_once){
        if (id > hazard_buckets[bucket]){
            ++bucket;
            hazard_rate -= 0.02;
        }
        start_of_data = counterparty_deals_infile.tellg();
        fx_count = 0;
        swap_count = 0;

        do{
            counterparty_deals_infile >> deal_id;
            if (deal_id<params.fx_num) ++fx_count;
            else ++swap_count;
            counterparty_deals_infile >> current_id;
        } while(current_id == id);
        counterparty_deals_infile.seekg(start_of_data,counterparty_deals_infile.beg);
        //std::cout << id << " " << fx_count << " " << swap_count << "\n";

        Counterparty cp(id, hazard_rate, fx_count, swap_count);
        do{
            counterparty_deals_infile >> deal_id;
            if (deal_id<params.fx_num){
                fx_details_infile >> fx_id;
                fx_details_infile >> notional;
                fx_details_infile >> position;
                cp.add_fx(fx_id, notional, position);
            }
            else {
                swap_details_infile >> swap_id;
                swap_details_infile >> denomination;
                swap_details_infile >> notional;
                swap_details_infile >> fixed_rate;
                swap_details_infile >> tenor;
                swap_details_infile >> position;
                cp.add_swap(swap_id, denomination, notional, fixed_rate, tenor, position);
            }
            ++deals_handled;
            counterparty_deals_infile >> current_id;
        } while(current_id == id);
        cp_vector.push_back(cp);
        ++id;
    }

    //int num_of_steps = 360*params.time_horizon/params.step_size;

    thrust::transform(cp_vector.begin(), cp_vector.end(), cva_vector.begin(), calculate_cva(params));
    //std::transform(cp_vector.begin(), cp_vector.end(), cva_vector.begin(), calculate_cva(params));


    //for (unsigned int i=0; i<cp_vector.size(); ++i){
    //    std::cout << "cva " << i+1 << " " << cp_vector[i].cva << "\n";
    //}

    for (unsigned int i=0; i<cva_vector.size(); ++i){
        std::cout << "cva " << i+1 << " " << cva_vector[i] << "\n";
    }


    counterparty_deals_infile.close();
    fx_details_infile.close();
    swap_details_infile.close();

    std::cout << "\n";


    return 0;
}

