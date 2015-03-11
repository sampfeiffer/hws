//seed
//code cleaning
//timing for different parts

#include <thrust/reduce.h>
#include <thrust/copy.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <omp.h>
#include <numeric>

#include "parameters.h"
#include "data_reader.h"
#include "functors.h"
#include "helper_functions.h"
#include "timing.h"

typedef thrust::device_vector<Fx> dvec_fx;
typedef dvec_fx *p_dvec_fx;
typedef thrust::device_vector<Swap> dvec_swap;
typedef dvec_swap *p_dvec_swap;

int main(int argc, char *argv[])
{
    // Start timing the program
    //clock_t program_start_time, mid_time, end_time;
    //clock_t mid_time, end_time;
    //program_start_time = clock();
    Timing program_timing("whole program");

    // helper functions
    thrust::host_vector<State> generate_paths(Parameters &params, int &num_of_steps);
    int gpu_info(int &num_gpus, Parameters &params);
    thrust::device_vector<int> cva_average_over_gpu(std::vector<p_cva_vec> &cva_vectors_std, int &deals_at_once, int &num_gpus);
    void convert_to_counterparties(std::vector<long> &total_cva, thrust::host_vector<float> &cva_vector_host, Parameters &params, const char* counterparty_deals_filename, bool is_fx);
    void print_results(std::vector<long> &total_cva, float &multiple);

    const char* parameters_filename="parameters.txt";
    const char* state0_filename="state0.txt";
    const char* counterparty_deals_filename="counterparty_deals.dat";

    // Get parameters and initial state of the world.
    Parameters params(parameters_filename, state0_filename);
    int num_of_steps = params.days_in_year*params.time_horizon/params.step_size;
    int num_gpus=0;
    float multiple = 1-params.recovery_rate;
    std::vector<p_dvec_fx> dvecs_fx;
    std::vector<p_dvec_swap> dvecs_swap;
    std::vector<p_cva_vec> cva_vectors_std;
    std::vector<long> total_cva(params.counterparty_num, 0);

    // Determine the number of CUDA capable GPUs. Calculate the number of deals to handle at a time.
    int deals_at_once = gpu_info(num_gpus, params);
    int paths_per_gpu = params.simulation_num/num_gpus; // Paths are split between the GPUs

    // Generate the simulation paths.
    thrust::device_vector<State> dpaths = generate_paths(params, num_of_steps);
    State* path_ptr = thrust::raw_pointer_cast(dpaths.data());

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

    Timing fx_timing("CVA for FX");
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
                              calculate_cva(num_of_steps, paths_per_gpu, path_ptr+k*params.fx_num/deals_at_once));
            cudaDeviceSynchronize();
        }

        // Find the average of the cva calculations over the different GPUs
        thrust::device_vector<int> cva_average = cva_average_over_gpu(cva_vectors_std, deals_at_once, num_gpus);
        thrust::copy(cva_average.begin(), cva_average.end(), cva_vector_host.begin()+k*deals_at_once);
    }

    // Convert from CVA for fx deals to CVA for counterparties
    convert_to_counterparties(total_cva, cva_vector_host, params, counterparty_deals_filename, true);

    //mid_time = clock();
    std::cout << "Finished FX deals\n";
    //std::cout << "Timing: FX CVA " << float(mid_time-program_start_time)/CLOCKS_PER_SEC << " seconds.\n";
    fx_timing.end_timing();

    //----------------------------------------------------------------------------------------------------
    // SWAPS

    Timing swap_timing("CVA for swaps");
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
            thrust::transform((*(dvecs_swap[cpu_thread_id])).begin(), (*(dvecs_swap[cpu_thread_id])).end(), (*(cva_vectors_std[cpu_thread_id])).begin(),
                              calculate_cva(num_of_steps, paths_per_gpu, path_ptr+k*params.fx_num/deals_at_once));
            cudaDeviceSynchronize();
        }

        // Find the average of the cva calculations over the different GPUs
        thrust::device_vector<int> cva_average = cva_average_over_gpu(cva_vectors_std, deals_at_once, num_gpus);
        thrust::copy(cva_average.begin(), cva_average.end(), cva_vector_host.begin()+k*deals_at_once);
    }

    // Convert from CVA for swaps to CVA for counterparties
    convert_to_counterparties(total_cva, cva_vector_host, params, counterparty_deals_filename, false);

    data.close_files();
    //end_time = clock();
    std::cout << "Finished Swap deals\n";
    //std::cout << "Timing: Swap CVA " << float(end_time-mid_time)/CLOCKS_PER_SEC << " seconds.\n";
    swap_timing.end_timing();

    // Results
    print_results(total_cva, multiple);

    //end_time = clock() - program_start_time;
    //std::cout << "Timing: Whole program " << float(end_time)/CLOCKS_PER_SEC << " seconds.\n";
    program_timing.end_timing();
    std::cout << "\n";
    return 0;
}
