//seed
//code cleaning
//timing for different parts

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/copy.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <omp.h>

#include <vector>
#include <numeric>
//#include <sys/time.h>
#include <time.h>

#include "parameters.h"
#include "data_reader.h"
#include "state.h"
#include "functors.h"

typedef thrust::device_vector<Fx> dvec_fx;
typedef dvec_fx *p_dvec_fx;
typedef thrust::device_vector<Swap> dvec_swap;
typedef dvec_swap *p_dvec_swap;
typedef thrust::device_vector<float> cva_vector;
typedef cva_vector *p_cva_vec;

int main(int argc, char *argv[])
{
    // Start timing the program
    clock_t program_start_time, mid_time, end_time;
    program_start_time = clock();

    void print_results(std::vector<long> &total_cva, float &multiple);
    int gpu_info(int &num_gpus, Parameters &params);
    thrust::device_vector<int> cva_average_over_gpu(std::vector<p_cva_vec> &cva_vectors_std, int R, int C);

    const char* parameters_filename="parameters.txt";
    const char* state0_filename="state0.txt";
    const char* counterparty_deals_filename="counterparty_deals.dat";
    std::ifstream counterparty_deals_infile;
    int cp_id=1, cp_id_read, deal_id_read, deals_at_once, num_gpus=0, R, C;
    float cva_temp=0;

//    typedef thrust::device_vector<Fx> dvec_fx;
//    typedef dvec_fx *p_dvec_fx;
//    typedef thrust::device_vector<Swap> dvec_swap;
//    typedef dvec_swap *p_dvec_swap;
//    typedef thrust::device_vector<float> cva_vector;
//    typedef cva_vector *p_cva_vec;

    std::vector<p_dvec_fx> dvecs_fx;
    std::vector<p_dvec_swap> dvecs_swap;
    std::vector<p_cva_vec> cva_vectors_std;
    std::vector<long> total_cva;

    // Get parameters and initial state of the world.
    Parameters params(parameters_filename, state0_filename);
    int num_of_steps = params.days_in_year*params.time_horizon/params.step_size;
    int total_deals = params.fx_num + params.swap_num;
    float multiple = 1-params.recovery_rate;

    // Determine the number of CUDA capable GPUs. Calculate the number of deals to handle at a time.
    deals_at_once = gpu_info(num_gpus, params);
    int simulations_per_gpu = params.simulation_num/num_gpus; // Paths are split between the GPUs

    thrust::host_vector<State> hpaths;
    for (int sim=0; sim<params.simulation_num; ++sim){
        State world_state(params);
        for (int i=0; i<num_of_steps; ++i){
            world_state.sim_next_step();
            hpaths.push_back(world_state);
        }
    }

    thrust::device_vector<State> dpaths = hpaths;
    //State* path_ptr = thrust::raw_pointer_cast(&dpaths[0]);
    State* path_ptr = thrust::raw_pointer_cast(dpaths.data());

    //std::cout << "time test " << path_ptr[0].time << "\n";

    R = deals_at_once; // number of rows
    C = num_gpus; // number of columns
    // initialize data on each GPU
    for(unsigned int i = 0; i < num_gpus; i++) {
        cudaSetDevice(i);
        p_dvec_fx temp = new dvec_fx(deals_at_once);
        dvecs_fx.push_back(temp);
        p_cva_vec temp2 = new cva_vector(deals_at_once);
        cva_vectors_std.push_back(temp2);
    }

    std::cout << "-----------------------------------------\n";

    //----------------------------------------------------------------------------------------------------
    // FX

    std::cout << "Starting FX deals\n";
    Data_reader data;
    std::vector<Fx> fx_vector_temp;
    thrust::host_vector<float> cva_vector_host(params.fx_num);
    for (int k=0; k<params.fx_num/deals_at_once; ++k){
        // Get fx deal data and copy onto each GPU
        fx_vector_temp.clear();
        data.get_next_data_fx(fx_vector_temp, deals_at_once);
        for (unsigned int i = 0; i < num_gpus; i++) {
            cudaSetDevice(i);
            thrust::copy(fx_vector_temp.begin(), fx_vector_temp.end(), (*(dvecs_fx[i])).begin());
        }

        // run as many CPU threads as there are CUDA devices
        // Calculate CVA on each GPU
        omp_set_num_threads(num_gpus);
        #pragma omp parallel
        {
            unsigned int cpu_thread_id = omp_get_thread_num();
            cudaSetDevice(cpu_thread_id);
            thrust::transform((*(dvecs_fx[cpu_thread_id])).begin(), (*(dvecs_fx[cpu_thread_id])).end(), (*(cva_vectors_std[cpu_thread_id])).begin(),
                              calculate_cva_fx(num_of_steps, simulations_per_gpu, path_ptr+k*params.fx_num/deals_at_once));
            cudaDeviceSynchronize();
        }

        // Find the average of the cva calculations over the different GPUs
        thrust::device_vector<int> cva_sum(R * C);
        for (size_t j=0; j<C; j++){
            for (size_t i=0; i<R; i++){
                cva_sum[i*num_gpus+j] = (*(cva_vectors_std[j]))[i];
            }
        }

        // allocate storage for row sums and indices
        thrust::device_vector<int> row_sums(R);
        thrust::device_vector<int> row_indices(R);

        // compute row sums by summing values with equal row indices
        thrust::reduce_by_key
            (thrust::make_transform_iterator(thrust::counting_iterator<int>(0), linear_index_to_row_index<int>(C)),
            thrust::make_transform_iterator(thrust::counting_iterator<int>(0), linear_index_to_row_index<int>(C)) + (R*C),
            cva_sum.begin(),
            row_indices.begin(),
            row_sums.begin(),
            thrust::equal_to<int>(),
            thrust::plus<int>());



        thrust::device_vector<int> divisor(deals_at_once);
        thrust::device_vector<int> cva_average(deals_at_once);
        thrust::fill(divisor.begin(), divisor.end(), num_gpus);
        thrust::transform(row_sums.begin(), row_sums.end(), divisor.begin(), cva_average.begin(), thrust::divides<int>()); //divide by the num of gpu's used to find the average.

        //thrust::device_vector<int> cva_average = cva_average_over_gpu(cva_vectors_std, R, C);

        thrust::copy(cva_average.begin(), cva_average.end(), cva_vector_host.begin()+k*deals_at_once);
    }

    counterparty_deals_infile.open(counterparty_deals_filename);
    if (!counterparty_deals_infile.is_open()){
        std::cout << "ERROR: counterparty_deals.dat file could not be opened. Exiting.\n";
        exit(1);
    }
    // Convert CVA for FX deals into CVA for counterparties
    counterparty_deals_infile >> cp_id_read;
    for (int i=0; i<total_deals; ++i){
        counterparty_deals_infile >> deal_id_read;
        if (deal_id_read <= params.fx_num){
            cva_temp += cva_vector_host[deal_id_read-1];
        }
        counterparty_deals_infile >> cp_id_read;
        if (cp_id_read > cp_id || counterparty_deals_infile.eof()){
            total_cva.push_back(cva_temp);
            cva_temp = 0;
            ++cp_id;
        }
    }

    mid_time = clock();
    std::cout << "Finished FX deals\n";
    std::cout << "Timing: FX CVA " << float(mid_time-program_start_time)/CLOCKS_PER_SEC << " seconds.\n";

    //----------------------------------------------------------------------------------------------------
    // SWAPS

    std::cout << "Starting Swap deals\n";

    // initialize data on each GPU
    for(unsigned int i = 0; i < num_gpus; i++) {
        cudaSetDevice(i);
        p_dvec_swap temp = new dvec_swap(deals_at_once);
        dvecs_swap.push_back(temp);
    }

    std::vector<Swap> swap_vector_temp;
    thrust::fill(cva_vector_host.begin(), cva_vector_host.end(), 0);
    for (int k=0; k<params.swap_num/deals_at_once; ++k){
        // Get swap deal data and copy onto each GPU
        swap_vector_temp.clear();
        data.get_next_data_swap(swap_vector_temp, deals_at_once);
        for (unsigned int i = 0; i < num_gpus; i++) {
            cudaSetDevice(i);
            thrust::copy(swap_vector_temp.begin(), swap_vector_temp.end(), (*(dvecs_swap[i])).begin());
        }

        // run as many CPU threads as there are CUDA devices
        // Calculate CVA on each GPU
        omp_set_num_threads(num_gpus);
        #pragma omp parallel
        {
            unsigned int cpu_thread_id = omp_get_thread_num();
            cudaSetDevice(cpu_thread_id);
            thrust::transform((*(dvecs_swap[cpu_thread_id])).begin(), (*(dvecs_swap[cpu_thread_id])).end(), (*(cva_vectors_std[cpu_thread_id])).begin(), calculate_cva_swap(num_of_steps, simulations_per_gpu, path_ptr+k*params.fx_num/deals_at_once));
            cudaDeviceSynchronize();
        }

        // Find the average of the cva calculations over the different GPUs
        thrust::device_vector<int> cva_sum(R * C);
        for (size_t j=0; j<C; j++){
            for (size_t i=0; i<R; i++){
                cva_sum[i*num_gpus+j] = (*(cva_vectors_std[j]))[i];
            }
        }

        // allocate storage for row sums and indices
        thrust::device_vector<int> row_sums(R);
        thrust::device_vector<int> row_indices(R);

        // compute row sums by summing values with equal row indices
        thrust::reduce_by_key
            (thrust::make_transform_iterator(thrust::counting_iterator<int>(0), linear_index_to_row_index<int>(C)),
            thrust::make_transform_iterator(thrust::counting_iterator<int>(0), linear_index_to_row_index<int>(C)) + (R*C),
            cva_sum.begin(),
            row_indices.begin(),
            row_sums.begin(),
            thrust::equal_to<int>(),
            thrust::plus<int>());

        thrust::device_vector<int> divisor(deals_at_once);
        thrust::device_vector<int> cva_average(deals_at_once);
        thrust::fill(divisor.begin(), divisor.end(), num_gpus);
        thrust::transform(row_sums.begin(), row_sums.end(), divisor.begin(), cva_average.begin(), thrust::divides<int>()); //divide by the num of gpu's used to find the average.
        thrust::copy(cva_average.begin(), cva_average.end(), cva_vector_host.begin()+k*deals_at_once);
    }

    // Convert CVA for swaps into CVA for counterparties
    cp_id=1;
    cva_temp=0;
    counterparty_deals_infile.clear(); //gets rid of eof flag
    counterparty_deals_infile.seekg(0, counterparty_deals_infile.beg);
    counterparty_deals_infile >> cp_id_read;
    for (int i=0; i<total_deals; ++i){
        counterparty_deals_infile >> deal_id_read;
        if (deal_id_read > params.fx_num){
            cva_temp += cva_vector_host[deal_id_read-1-params.fx_num];
        }
        counterparty_deals_infile >> cp_id_read;
        if (cp_id_read > cp_id || counterparty_deals_infile.eof()){
            total_cva[cp_id-1] += cva_temp;
            cva_temp = 0;
            ++cp_id;
        }
    }

    data.close_files();
    counterparty_deals_infile.close();
    end_time = clock();
    std::cout << "Finished Swap deals\n";
    std::cout << "Timing: Swap CVA " << float(end_time-mid_time)/CLOCKS_PER_SEC << " seconds.\n";

    // Results
    print_results(total_cva, multiple);

    end_time = clock() - program_start_time;
    std::cout << "Timing: Whole program " << float(end_time)/CLOCKS_PER_SEC << " seconds.\n";
    std::cout << "\n";
    return 0;
}

// Print results
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

thrust::device_vector<int> cva_average_over_gpu(std::vector<p_cva_vec> &cva_vectors_std, int R, int C)
{
    // Find the average of the cva calculations over the different GPUs
    thrust::device_vector<int> cva_sum(R * C);
    for (size_t j=0; j<C; j++){
        for (size_t i=0; i<R; i++){
            cva_sum[i*C+j] = (*(cva_vectors_std[j]))[i];
        }
    }

    // allocate storage for row sums and indices
    thrust::device_vector<int> row_sums(R);
    thrust::device_vector<int> row_indices(R);

    // compute row sums by summing values with equal row indices
    thrust::reduce_by_key
        (thrust::make_transform_iterator(thrust::counting_iterator<int>(0), linear_index_to_row_index<int>(C)),
        thrust::make_transform_iterator(thrust::counting_iterator<int>(0), linear_index_to_row_index<int>(C)) + (R*C),
        cva_sum.begin(),
        row_indices.begin(),
        row_sums.begin(),
        thrust::equal_to<int>(),
        thrust::plus<int>());

    thrust::device_vector<int> divisor(C);
    thrust::device_vector<int> cva_average(C);
    thrust::fill(divisor.begin(), divisor.end(), R);
    thrust::transform(row_sums.begin(), row_sums.end(), divisor.begin(), cva_average.begin(), thrust::divides<int>()); //divide by the num of gpu's used to find the average.

    return cva_average;
}

