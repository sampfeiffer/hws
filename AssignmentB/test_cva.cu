//multiple gpus
//path creation off of device. pass to device
//multiply by recovery

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <omp.h>
#include <stdlib.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <time.h>
#include <sys/time.h>

#include <vector>
#include <time.h>
#include <numeric>
#include "parameters.h"
#include "data_reader.h"
#include "state.h"

// convert a linear index to a row index
template <typename T>
struct linear_index_to_row_index : public thrust::unary_function<T,T>
{
    T C; // number of columns
    __host__ __device__
    linear_index_to_row_index(T C) : C(C) {}
    __host__ __device__
    T operator()(T i)
    {
    return i / C;
    }
};

struct calculate_cva_fx{
    Parameters params;
    int num_of_steps;

    calculate_cva_fx(Parameters params_, int num_of_steps_) : params(params_), num_of_steps(num_of_steps_)
    {}
    __device__ __host__
    float operator()(Fx &fx) {
        float cva=0;
        float prob_default;
        State world_state(params);
        for (int i=0; i<num_of_steps; ++i){
            world_state.sim_next_step();
            prob_default = std::exp(-fx.hazard_rate*(world_state.time-1)/float(360.0)) - std::exp(-fx.hazard_rate*world_state.time/float(360.0));
            cva += world_state.cva_disc_factor * prob_default * max(fx.value(world_state.fx_rate),float(0.0));
        }
        cva *= 1-params.recovery_rate;
        return cva;
    }
};

struct calculate_cva_swap{
    Parameters params;
    int num_of_steps;

    calculate_cva_swap(Parameters params_, int num_of_steps_) : params(params_), num_of_steps(num_of_steps_)
    {}
    __device__ __host__
    float operator()(Swap &sw) {
        float cva=0;
        float prob_default;
        State world_state(params);
        for (int i=0; i<num_of_steps; ++i){
            world_state.sim_next_step();
            prob_default = std::exp(-sw.hazard_rate*(world_state.time-1)/float(360.0)) - std::exp(-sw.hazard_rate*world_state.time/float(360.0));
            cva += world_state.cva_disc_factor * prob_default * max(sw.value(world_state),float(0.0));
        }
        cva *= 1-params.recovery_rate;
        return cva;
    }
};

int main(int argc, char *argv[])
{
    clock_t program_start_time, end_time;
    program_start_time = clock();

    const char* parameters_filename="parameters.txt";
    const char* state0_filename="state0.txt";

    // Get parameters and initial state of the world.
    Parameters params(parameters_filename, state0_filename);
    int num_of_steps = params.days_in_year*params.time_horizon/params.step_size;

    // determine the number of CUDA capable GPUs
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
        printf("   %d: %s\n", i, dprop.name);
    }
    //int simulations_per_gpu = params.simulation_num/num_gpus;


    const char* counterparty_deals_filename="counterparty_deals.dat";
    std::ifstream counterparty_deals_infile;
    counterparty_deals_infile.open(counterparty_deals_filename);
    if (!counterparty_deals_infile.is_open()){
        std::cout << "ERROR: counterparty_deals.dat file could not be opened. Exiting.\n";
        exit(1);
    }


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
        omp_set_num_threads(num_gpus);  // create as many CPU threads as there are CUDA devices
        #pragma omp parallel
        {
            unsigned int cpu_thread_id = omp_get_thread_num();
            cudaSetDevice(cpu_thread_id);
            thrust::transform((*(dvecs[cpu_thread_id])).begin(), (*(dvecs[cpu_thread_id])).end(), (*(cva_vectors_std[cpu_thread_id])).begin(), calculate_cva_fx(params, num_of_steps));
            cudaDeviceSynchronize();
        }

        int R = params.deals_at_once; // number of rows
        int C = num_gpus; // number of columns
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
        thrust::transform(cva_sum.begin(), cva_sum.end(), divisor.begin(), cva_average.begin(), thrust::divides<int>()); //divide by the num of gpu's used to find the average.
        thrust::copy(cva_average.begin(), cva_average.end(), cva_vector_host.begin()+k*params.deals_at_once);

    }

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
        if (cp_id_read > cp_id){
            total_cva.push_back(cva_temp);
            cva_temp = 0;
            ++cp_id;
        }
    }

    std::cout << "cva " << std::accumulate(total_cva.begin(), total_cva.end(), 0) << "\n";

    data.close_files();
    counterparty_deals_infile.close();

    end_time = clock() - program_start_time;
    std::cout << "Timing: whole program " << float(end_time)/CLOCKS_PER_SEC << " seconds.\n";
    std::cout << "\n";
    return 0;
}

