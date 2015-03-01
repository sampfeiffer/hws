#include <vector>
#include "parameters.h"
#include "counterparty.h"
#include "state.h"

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

    int num_of_steps = 360*params.time_horizon/params.step_size;

    // Generate a state of the world that will be changed through time
    State world_state(params);
    std::vector<State> state_vector;
    state_vector.push_back(world_state);
    for (int i=0; i<num_of_steps; ++i){
        world_state.sim_next_step();
        State next_state = world_state;
        state_vector.push_back(next_state);
    }

    // Calculate CVA
    for (unsigned int cp=0; cp<cp_vector.size(); ++cp){
        // CVA for fx
        for (unsigned int fx=0; fx<cp_vector[cp].fx_deals.size(); ++fx){
            for (unsigned int i=0; i<state_vector.size(); ++i){
                cp_vector[cp].cva += state_vector[i].cva_disc_factor * cp_vector[cp].prob_default(state_vector[i].time)
                                     * std::max(cp_vector[cp].fx_deals[fx].value(state_vector[i].fx_rate_beg, state_vector[i].fx_rate),0.0);
            }
        }
        // CVA for swaps
        for (unsigned int sw=0; sw<cp_vector[cp].swap_deals.size(); ++sw){
            for (unsigned int i=0; i<state_vector.size(); ++i){
                cp_vector[cp].cva += state_vector[i].cva_disc_factor * cp_vector[cp].prob_default(state_vector[i].time)
                                     * std::max(cp_vector[cp].swap_deals[sw].value(state_vector[i]),0.0);
            }
        }
        cp_vector[cp].cva *= 1-params.recovery_rate;
    }

    for (unsigned int i=0; i<cp_vector.size(); ++i){
        std::cout << "cva " << i+1 << " " << cp_vector[i].cva << "\n";
    }

    counterparty_deals_infile.close();
    fx_details_infile.close();
    swap_details_infile.close();

    std::cout << "\n";

    //std::cout << "test " << state_vector[0].fx_rate_beg << "\n";

    return 0;
}
