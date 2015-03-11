#ifndef HELPER_FUNCTIONS_INCLUDED
#define HELPER_FUNCTIONS_INCLUDED

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <iostream>
#include <vector>

#include "state.h"

typedef thrust::device_vector<float> cva_vector;
typedef cva_vector *p_cva_vec;

// Determine the number of GPUs, their properties, and return the number of deals to handle at once
int gpu_info(int &num_gpus, Parameters &params)
{
    int deals_at_once;
    // Determine the number of CUDA capable GPUs.
    cudaGetDeviceCount(&num_gpus);
    if (num_gpus > 1) --num_gpus; // I believe it counts a cpu also...
    if (num_gpus < 1)
    {
        printf("no CUDA capable devices were detected\n");
        return 1;
    }
    printf("number of host CPUs:\t%d\n", omp_get_num_procs());
    printf("number of CUDA devices:\t%d\n", num_gpus);
    for (int i = 0; i < num_gpus; i++)
    {
        cudaDeviceProp dprop;
        cudaGetDeviceProperties(&dprop, i);
        printf("   %d: %s. Memory available: %dMB\n", i+1, dprop.name, int(dprop.totalGlobalMem / (1024 * 1024)));
        deals_at_once = dprop.totalGlobalMem/100;
    }

    deals_at_once = min(deals_at_once, params.fx_num);
    if (params.fx_num % deals_at_once != 0) deals_at_once = (params.fx_num/deals_at_once) - (params.fx_num%deals_at_once); //Make sure it divides evenly;
    std::cout << "Deals handled at once " << deals_at_once << "\n";
    return deals_at_once;
}

// Generate all the simulation paths
thrust::host_vector<State> generate_paths(Parameters &params, int &num_of_steps)
{
    thrust::host_vector<State> hpaths;
    for (int sim=0; sim<params.simulation_num; ++sim){
        State world_state(params);
        for (int i=0; i<num_of_steps; ++i){
            world_state.sim_next_step();
            hpaths.push_back(world_state);
        }
    }

    return hpaths;
}

// Calculate the CVA average over the paths from different GPUs.
thrust::device_vector<int> cva_average_over_gpu(std::vector<p_cva_vec> &cva_vectors_std, int &deals_at_once, int &num_gpus)
{
    // Get the vector of CVA vectors from the different GPUs into one vector.
    thrust::device_vector<int> cva_sum(deals_at_once * num_gpus);
    for (size_t j=0; j<num_gpus; j++){
        for (size_t i=0; i<deals_at_once; i++){
            cva_sum[j*num_gpus+i] = (*(cva_vectors_std[j]))[i];
        }
    }

    // allocate storage for row sums and indices
    thrust::device_vector<int> row_sums(deals_at_once);
    thrust::device_vector<int> row_indices(deals_at_once);

    // add up the CVA for the same deal over different GPUs
    thrust::reduce_by_key
        (thrust::make_transform_iterator(thrust::counting_iterator<int>(0), linear_index_to_row_index<int>(deals_at_once)),
        thrust::make_transform_iterator(thrust::counting_iterator<int>(0), linear_index_to_row_index<int>(deals_at_once)) + (deals_at_once*num_gpus),
        cva_sum.begin(),
        row_indices.begin(),
        row_sums.begin(),
        thrust::equal_to<int>(),
        thrust::plus<int>());

    // divide by the number of GPUs to get the average over the paths
    thrust::for_each(row_sums.begin(), row_sums.end(), divide_by<int>(num_gpus));
    return row_sums;
}

// Convert CVA for each deal into CVA for each counterparty
void convert_to_counterparties(std::vector<long> &total_cva, thrust::host_vector<float> &cva_vector_host, Parameters &params, const char* counterparty_deals_filename, bool is_fx)
{
    int cp_id=1, cp_id_read, deal_id_read;
    int total_deals = params.fx_num + params.swap_num;
    float cva_temp=0;
    std::ifstream counterparty_deals_infile;

    counterparty_deals_infile.open(counterparty_deals_filename);
    if (!counterparty_deals_infile.is_open()){
        std::cout << "ERROR: " << counterparty_deals_filename << " file could not be opened. Exiting.\n";
        exit(1);
    }

    counterparty_deals_infile >> cp_id_read;
    for (int i=0; i<total_deals; ++i){
        counterparty_deals_infile >> deal_id_read;
        if (is_fx && deal_id_read <= params.fx_num){
            cva_temp += cva_vector_host[deal_id_read-1];
        }
        else if (!is_fx && deal_id_read > params.fx_num){
            cva_temp += cva_vector_host[deal_id_read-1-params.fx_num];
        }
        counterparty_deals_infile >> cp_id_read;
        if (cp_id_read > cp_id || counterparty_deals_infile.eof()){
            total_cva[cp_id-1] += cva_temp;
            cva_temp = 0;
            ++cp_id;
        }
    }
    counterparty_deals_infile.close();
}

// Print CVA results
void print_results(std::vector<long> &total_cva, float &multiple)
{
    const char* cva_output_filename="cva_output.txt";
    std::ofstream cva_output_outfile;
    cva_output_outfile.open(cva_output_filename);
    if (!cva_output_outfile.is_open()){
        std::cout << "ERROR: cva_output.txt file could not be opened. Exiting.\n";
        exit(1);
    }
    cva_output_outfile << "Counterparty  CVA\n";
    for (unsigned int cp=0; cp<total_cva.size(); ++cp){
        cva_output_outfile << cp+1 << " " << multiple*total_cva[cp] << "\n";
    }
    cva_output_outfile << "\nGrand Total Bank CVA " << multiple*std::accumulate(total_cva.begin(), total_cva.end(), 0) << "\n";
    cva_output_outfile.close();

    std::cout << "-----------------------------------------\n";
    std::cout << "\nGrand Total Bank CVA " << multiple*std::accumulate(total_cva.begin(), total_cva.end(), 0) << "\n";
}

#endif // HELPER_FUNCTIONS_INCLUDED
