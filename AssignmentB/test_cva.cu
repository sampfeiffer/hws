// calculate how many deals i can read in to one gpu.
// create a device_vector of the appropriate amount of counterparties
// run the simulations and cva calculator on the vector of counterparties

#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <time.h>
#include <sys/time.h>

#include <vector>
#include "parameters.h"
#include "counterparty.h"
#include "state.h"

#define DSIZE 1000

using namespace std;

int main(int argc, char *argv[])
{
    clock_t program_start_time, end_time;
    program_start_time = clock();

    const char* parameters_filename="parameters.txt";
    const char* state0_filename="state0.txt";
    const char* hazard_buckets_filename="hazard_buckets.dat";
    const char* counterparty_deals_filename="counterparty_deals.dat";
    const char* fx_details_filename="fx_details.dat";
    const char* swap_details_filename="swap_details.dat";
    std::ifstream counterparty_deals_infile, fx_details_infile, swap_details_infile, hazard_buckets_infile;

    // Get parameters and initial state of the world.
    Parameters params(parameters_filename, state0_filename);
    //params.print();

    // Get the list of hazard rate bucket endpoints
    int hazard_buckets[5];
    hazard_buckets_infile.open(hazard_buckets_filename);
    if (!hazard_buckets_infile.is_open()){
        std::cout << "ERROR: hazard_buckets.dat file could not be opened. Exiting.\n";
        exit(1);
    }
    for (int i=0; i<5; ++i) hazard_buckets_infile >> hazard_buckets[i];
    hazard_buckets_infile.close();


    // Open the counterparty deals and deal details
    counterparty_deals_infile.open(counterparty_deals_filename);
    if (!counterparty_deals_infile.is_open()){
        std::cout << "ERROR: counterparty_deals.dat file could not be opened. Exiting.\n";
        exit(1);
    }
    fx_details_infile.open(fx_details_filename);
    if (!fx_details_infile.is_open()){
        std::cout << "ERROR: fx_details.dat file could not be opened. Exiting.\n";
        exit(1);
    }
    swap_details_infile.open(swap_details_filename);
    if (!swap_details_infile.is_open()){
        std::cout << "ERROR: swap_details.dat file could not be opened. Exiting.\n";
        exit(1);
    }

    thrust::device_vector<Counterparty> cp_vector;
    //thrust::host_vector<Counterparty> cp_vector;
    //std::vector<Counterparty> cp_vector;

    // Read deals into memory
    int current_id=1, deal_id, id=1, deals_handled=0, bucket=0;
    float hazard_rate=0.10;

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

    int num_of_steps = params.days_in_year*params.time_horizon/params.step_size;

//    timeval t1, t2;
    int num_gpus = 0;   // number of CUDA GPUs

    printf("%s Starting...\n\n", argv[0]);

    // determine the number of CUDA capable GPUs
    cudaGetDeviceCount(&num_gpus);

    if (num_gpus < 1)
    {
        printf("no CUDA capable devices were detected\n");
        return 1;
    }

    // display CPU and GPU configuration
    printf("number of host CPUs:\t%d\n", omp_get_num_procs());
    printf("number of CUDA devices:\t%d\n", num_gpus);

    for (int i = 0; i < num_gpus; i++)
    {
        cudaDeviceProp dprop;
        cudaGetDeviceProperties(&dprop, i);
        printf("   %d: %s\n", i, dprop.name);
    }

    printf("initialize data\n");


    // initialize data
    typedef thrust::device_vector<int> dvec;
    typedef dvec *p_dvec;
    std::vector<p_dvec> dvecs;

    for(unsigned int i = 0; i < num_gpus; i++) {
        cudaSetDevice(i);
        p_dvec temp = new dvec(DSIZE);
        dvecs.push_back(temp);
    }

    thrust::host_vector<int> data(DSIZE);
    thrust::generate(data.begin(), data.end(), rand);

    // copy data
    for (unsigned int i = 0; i < num_gpus; i++) {
        cudaSetDevice(i);
        thrust::copy(data.begin(), data.end(), (*(dvecs[i])).begin());
    }

    printf("start sort\n");
//    gettimeofday(&t1,NULL);

    // run as many CPU threads as there are CUDA devices
    omp_set_num_threads(num_gpus);  // create as many CPU threads as there are CUDA devices
    #pragma omp parallel
    {
        unsigned int cpu_thread_id = omp_get_thread_num();
        cudaSetDevice(cpu_thread_id);
        thrust::sort((*(dvecs[cpu_thread_id])).begin(), (*(dvecs[cpu_thread_id])).end());
        cudaDeviceSynchronize();
    }
//    gettimeofday(&t2,NULL);
//    printf("finished\n");
//    unsigned long et = ((t2.tv_sec * 1000000)+t2.tv_usec) - ((t1.tv_sec * 1000000) + t1.tv_usec);
//    if (cudaSuccess != cudaGetLastError())
//        printf("%s\n", cudaGetErrorString(cudaGetLastError()));
//    printf("sort time = %fs\n", (float)et/(float)(1000000));
//    // check results
//    thrust::host_vector<int> result(DSIZE);
//    thrust::sort(data.begin(), data.end());
//    for (int i = 0; i < num_gpus; i++)
//    {
//        cudaSetDevice(i);
//        thrust::copy((*(dvecs[i])).begin(), (*(dvecs[i])).end(), result.begin());
//        for (int j = 0; j < DSIZE; j++)
//          if (data[j] != result[j]) { printf("mismatch on device %d at index %d, host: %d, device: %d\n", i, j, data[j], result[j]); return 1;}
//    }
    printf("Success\n");



    counterparty_deals_infile.close();
    fx_details_infile.close();
    swap_details_infile.close();

    end_time = clock() - program_start_time;
    std::cout << "Timing: whole program " << float(end_time)/CLOCKS_PER_SEC << " seconds.\n";

    std::cout << "\n";

    return 0;
}
