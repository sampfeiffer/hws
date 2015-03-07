#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/copy.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <omp.h>

#include <vector>
#include <numeric>
#include <sys/time.h>
#include <time.h>

#include "parameters.h"
#include "data_reader.h"
#include "state.h"
#include "functors.h"

int main(int argc, char *argv[])
{
    // Strat timing the program
    clock_t program_start_time, mid_time, end_time;
    program_start_time = clock();

    const char* parameters_filename="parameters.txt";
    const char* state0_filename="state0.txt";

    // Get parameters and initial state of the world.
    Parameters params(parameters_filename, state0_filename);
    int num_of_steps = params.days_in_year*params.time_horizon/params.step_size;

    // Determine the number of CUDA capable GPUs.
    int num_gpus = 0;
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
        printf("   %d: %s. Memory available: %d\n", i, dprop.name, dprop.totalGlobalMem / (1024. * 1024.));
    }

    int simulations_per_gpu = params.simulation_num/num_gpus; // Paths are split between the GPUs


    const char* counterparty_deals_filename="counterparty_deals.dat";
    std::ifstream counterparty_deals_infile;
    counterparty_deals_infile.open(counterparty_deals_filename);
    if (!counterparty_deals_infile.is_open()){
        std::cout << "ERROR: counterparty_deals.dat file could not be opened. Exiting.\n";
        exit(1);
    }

    int R = params.deals_at_once; // number of rows
    int C = num_gpus; // number of columns
    // initialize data
    typedef thrust::device_vector<Fx> dvec;
    typedef thrust::device_vector<float> cva_vector;
    typedef dvec *p_dvec;
    typedef cva_vector *p_cva_vec;
    std::vector<p_dvec> dvecs;
    std::vector<p_cva_vec> cva_vectors_std;
    for(unsigned int i = 0; i < num_gpus; i++) {
        cudaSetDevice(i);
        p_dvec temp = new dvec(params.deals_at_once);
        dvecs.push_back(temp);
        p_cva_vec temp2 = new cva_vector(params.deals_at_once);
        cva_vectors_std.push_back(temp2);
    }

    std::cout << "-----------------------------------------\n";
    std::cout << "Starting FX deals\n";
    Data_reader data;
    std::vector<Fx> fx_vector_temp;
    thrust::host_vector<float> cva_vector_host(params.fx_num);
    for (int k=0; k<params.fx_num/params.deals_at_once; ++k){
        // Get fx deal data
        fx_vector_temp.clear();
        data.get_next_data_fx(fx_vector_temp, params);
        for (unsigned int i = 0; i < num_gpus; i++) {
            cudaSetDevice(i);
            thrust::copy(fx_vector_temp.begin(), fx_vector_temp.end(), (*(dvecs[i])).begin());
        }

        // run as many CPU threads as there are CUDA devices
        omp_set_num_threads(num_gpus);
        #pragma omp parallel
        {
            unsigned int cpu_thread_id = omp_get_thread_num();
            cudaSetDevice(cpu_thread_id);
            thrust::transform((*(dvecs[cpu_thread_id])).begin(), (*(dvecs[cpu_thread_id])).end(), (*(cva_vectors_std[cpu_thread_id])).begin(), calculate_cva_fx(params, num_of_steps, simulations_per_gpu));
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

        thrust::device_vector<int> divisor(params.deals_at_once);
        thrust::device_vector<int> cva_average(params.deals_at_once);
        thrust::fill(divisor.begin(), divisor.end(), num_gpus);
        thrust::transform(row_sums.begin(), row_sums.end(), divisor.begin(), cva_average.begin(), thrust::divides<int>()); //divide by the num of gpu's used to find the average.
        thrust::copy(cva_average.begin(), cva_average.end(), cva_vector_host.begin()+k*params.deals_at_once);

    }

    // Convert CVA for FX deals into CVA for counterparties
    int total_deals = params.fx_num + params.swap_num;
    int cp_id=1, cp_id_read, deal_id_read;
    float cva_temp=0;
    std::vector<float> total_cva;

    counterparty_deals_infile >> cp_id_read;
    for (int i=0; i<total_deals; ++i){
        counterparty_deals_infile >> deal_id_read;
        if (deal_id_read <= params.fx_num){
            cva_temp += cva_vector_host[deal_id_read-1];
        }
        counterparty_deals_infile >> cp_id_read;
        if (cp_id_read > cp_id || counterparty_deals_infile.eof()){
            std::cout << "cp_id " << cp_id << "\n";
            total_cva.push_back(cva_temp);
            cva_temp = 0;
            ++cp_id;
        }
    }

    mid_time = clock();
    std::cout << "Finished FX deals\n";
    std::cout << "Timing: FX CVA " << float(mid_time-program_start_time)/CLOCKS_PER_SEC << " seconds.\n";
    std::cout << "Starting Swap deals\n";

    // initialize data
    typedef thrust::device_vector<Swap> dvec_swap;
    typedef dvec_swap *p_dvec_swap;
    std::vector<p_dvec_swap> dvecs_swap;
    for(unsigned int i = 0; i < num_gpus; i++) {
        cudaSetDevice(i);
        p_dvec_swap temp = new dvec_swap(params.deals_at_once);
        dvecs_swap.push_back(temp);
    }

    std::vector<Swap> swap_vector_temp;
    thrust::fill(cva_vector_host.begin(), cva_vector_host.end(), 0);
    for (int k=0; k<params.swap_num/params.deals_at_once; ++k){
        // Get swap deal data
        swap_vector_temp.clear();
        data.get_next_data_swap(swap_vector_temp, params);

        for (unsigned int i = 0; i < num_gpus; i++) {
            cudaSetDevice(i);
            thrust::copy(swap_vector_temp.begin(), swap_vector_temp.end(), (*(dvecs_swap[i])).begin());
        }

        // run as many CPU threads as there are CUDA devices
        omp_set_num_threads(num_gpus);
        #pragma omp parallel
        {
            unsigned int cpu_thread_id = omp_get_thread_num();
            cudaSetDevice(cpu_thread_id);
            thrust::transform((*(dvecs_swap[cpu_thread_id])).begin(), (*(dvecs_swap[cpu_thread_id])).end(), (*(cva_vectors_std[cpu_thread_id])).begin(), calculate_cva_swap(params, num_of_steps, simulations_per_gpu));
            cudaDeviceSynchronize();
        }

        // initialize data
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

        thrust::device_vector<int> divisor(params.deals_at_once);
        thrust::device_vector<int> cva_average(params.deals_at_once);
        thrust::fill(divisor.begin(), divisor.end(), num_gpus);
        thrust::transform(row_sums.begin(), row_sums.end(), divisor.begin(), cva_average.begin(), thrust::divides<int>()); //divide by the num of gpu's used to find the average.
        thrust::copy(cva_average.begin(), cva_average.end(), cva_vector_host.begin()+k*params.deals_at_once);
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
            std::cout << "cp_id " << cp_id << "\n";
            total_cva[cp_id-1] += cva_temp;
            cva_temp = 0;
            ++cp_id;
        }
    }

    end_time = clock();
    std::cout << "Finished Swap deals\n";
    std::cout << "Timing: Swap CVA " << float(end_time-mid_time)/CLOCKS_PER_SEC << " seconds.\n";

    std::cout << "-----------------------------------------\n";
    std::cout << "Counterparty  CVA\n";
    float multiple = 1-params.recovery_rate;
    for (unsigned int cp=1; cp<total_cva.size(); ++cp){
        std::cout << cp << " " << multiple*total_cva[cp-1] << "\n";
    }


    std::cout << "\nGrand Total Bank CVA " << multiple*std::accumulate(total_cva.begin(), total_cva.end(), 0) << "\n";

    data.close_files();
    counterparty_deals_infile.close();

    end_time = clock() - program_start_time;
    std::cout << "Timing: whole program " << float(end_time)/CLOCKS_PER_SEC << " seconds.\n";
    std::cout << "\n";
    return 0;
}

