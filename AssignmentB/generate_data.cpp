#include <random>
#include "parameters.h"
#include <cmath>

void deal_distribution(Parameters &params)
{
    std::string sizes_filename="hazard_buckets.txt", counterparty_deals_filename="counterparty_deals.txt";
    std::ofstream sizes, counterparty_deals;
    const int buckets = 5;
    int bucket_size[buckets], fx_dist[buckets], swap_dist[buckets];
    std::default_random_engine generator;
    std::uniform_real_distribution<double> uniform(0.0,1.0);

    // Assign bucket sizes
    int bucket_total = 0;
    std::binomial_distribution<int> distribution(params.counterparty_num,0.2);
    for (int i=0; i<buckets-1; ++i){
        bucket_size[i] = distribution(generator)+bucket_total;
        bucket_total = bucket_size[i];
    }
    bucket_size[buckets-1] = params.counterparty_num;

    sizes.open(sizes_filename);
    if (!sizes.is_open()){
        std::cout << "ERROR: " << sizes_filename << " sizes.txt file could not be opened. Exiting.\n";
        exit(1);
    }
    for (int i=0; i<buckets; ++i){
        sizes << bucket_size[i] << " ";
    }
    sizes.close();

    //Put aside a mix of counterparty_num (1,000,000) fx and swaps to
    //make sure that each counterparty has a least one deal.
    std::binomial_distribution<int> distribution_aside(params.counterparty_num,0.5);
    int fx_aside = distribution_aside(generator);
    int swap_aside = params.counterparty_num - fx_aside;
    double fx_cutoff = double(fx_aside)/params.counterparty_num;



    //Assign number of fx and swaps in each bucket.
    int fx_total = params.fx_num-fx_aside;
    int swap_total = params.swap_num-swap_aside;
    for(int i=0; i<buckets-1; ++i){
        std::binomial_distribution<int> distribution2(params.fx_num,pow(2,i)/31.0);
        std::binomial_distribution<int> distribution3(params.swap_num,pow(2,i)/31.0);
        fx_dist[i] = distribution2(generator);
        swap_dist[i] = distribution3(generator);
        fx_total -= fx_dist[i];
        swap_total -= swap_dist[i];
    }
    fx_dist[buckets-1] = fx_total;
    swap_dist[buckets-1] = swap_total;


    int num_of_deals, fx_start_size, swap_start_size;
    int counterparty_id = 1, fx_id = 1, swap_id = params.fx_num+1;


    counterparty_deals.open(counterparty_deals_filename);
    if (!counterparty_deals.is_open()){
        std::cout << "ERROR: " << counterparty_deals_filename << " counterparty_deals.txt file could not be opened. Exiting.\n";
        exit(1);
    }

    // In each bucket...
    for (int i=0; i<buckets; ++i){
        // In each counterparty...
        fx_start_size = fx_dist[i];
        swap_start_size = swap_dist[i];
        for (int j=0; j<bucket_size[i]; ++j){
            // FX deals
            if (fx_dist[i] > 0){
                if (j<bucket_size[i]-1){
                    std::binomial_distribution<int> distribution_fx(fx_start_size,1.0/bucket_size[i]);
                    num_of_deals = std::min(distribution_fx(generator),fx_dist[i]);
                }
                else {
                    num_of_deals = fx_dist[i];
                }
                for (int k=0; k<num_of_deals; ++k){
                    //std::cout << counterparty_id << " " << fx_id << " bucket" << i+1 << " fxleft" << fx_dist[i] << "\n";
                    counterparty_deals << counterparty_id << " " << fx_id << "\n";
                    ++fx_id;
                    --fx_dist[i];
                }
            }

            // Figure out if the single necessary deal is an fx or swap.
            if (!swap_aside || (uniform(generator) < fx_cutoff && fx_aside)){
                //std::cout << counterparty_id << " " << fx_id << " bucket" << i+1 << " fxleft" << "ASIDE" << "\n";
                counterparty_deals << counterparty_id << " " << fx_id << "\n";
                ++fx_id;
                --fx_aside;
            }
            else{
                //std::cout << counterparty_id << " " << swap_id << " bucket" << i+1 << " swapleft" << "ASIDE" << "\n";
                counterparty_deals << counterparty_id << " " << swap_id << "\n";
                ++swap_id;
                --swap_aside;
            }

            // Swap deals
            if (swap_dist[i] > 0){
                if (j<bucket_size[i]-1){
                    std::binomial_distribution<int> distribution_swap(swap_start_size,1.0/bucket_size[i]);
                    num_of_deals = std::min(distribution_swap(generator), swap_dist[i]);
                }
                else {
                    num_of_deals = swap_dist[i];
                }
                for (int k=0; k<num_of_deals; ++k){
                    //std::cout << counterparty_id << " " << swap_id << " bucket" << i+1 << " swapleft" << swap_dist[i] << "\n";
                    counterparty_deals << counterparty_id << " " << swap_id << "\n";
                    ++swap_id;
                    --swap_dist[i];
                }
            }

            ++counterparty_id;
        }

    }

    counterparty_deals.close();
}

void assign_deal_details(Parameters &params)
{
    std::string fx_details_filename="fx_details.txt", swap_details_filename="swap_details.txt";
    std::ofstream fx_details, swap_details;
    fx_details.open(fx_details_filename);
    if (!fx_details.is_open()){
        std::cout << "ERROR: fx_details.txt file could not be opened. Exiting.\n";
        exit(1);
    }

    swap_details.open(swap_details_filename);
    if (!swap_details.is_open()){
        std::cout << "ERROR: swap_details.txt file could not be opened. Exiting.\n";
        exit(1);
    }

    // Randomly assign fx and swap parameters according to the specification
    for (int fx_id=1; fx_id<=params.fx_num; ++fx_id){
        fx_details << fx_id << " " << (rand()%400000) + 800000 << " " << ((rand()%10<6)?"l":"s") << "\n";
    }
    for (int swap_id=params.fx_num+1; swap_id<=params.fx_num+params.swap_num; ++swap_id){
        swap_details << swap_id << " " << ((rand()%2)?"u":"e") << " " << (rand()%400000) + 800000 << " " << ((rand()%7)+2)/100.0 << " " << ((rand()%20<9)?"l":"s") << "\n";
    }

    fx_details.close();
    swap_details.close();
}

int main(int argc, char *argv[])
{
    srand (time(NULL));
    std::string parameters_filename = "parameters.txt";

    Parameters params(parameters_filename);
    params.print();

    deal_distribution(params);
    assign_deal_details(params);

    std::cout << "\n";
    return 0;
}
