#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/random/linear_congruential_engine.h>
#include <thrust/random/normal_distribution.h>

#include <vector>
#include "parameters.h"
#include "test_counterparty.h"

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


    //thrust::device_vector<int> X(10);

    // Read deals into memory
    int current_id=1, deal_id, id=1, deals_handled=0, bucket=0;
    float hazard_rate=0.10;
    //Counterparty cp_vector(id,hazard_rate);
    thrust::device_vector<Counterparty> cp_vector;
    std::string deal_text;
    counterparty_deals_infile >> deal_id;





    counterparty_deals_infile.close();
    fx_details_infile.close();
    swap_details_infile.close();

    std::cout << "\n";

    //std::cout << "test " << state_vector[0].fx_rate_beg << "\n";

    thrust::minstd_rand rng;
    thrust::random::normal_distribution<float> dist(0.0f, 1.0f);
    std::cout << dist(rng) << std::endl;

    return 0;
}

