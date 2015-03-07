//turn cva for each deal into cva for each counterparty
//multiple gpus
//path creation off of device. pass to device

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <vector>
#include <time.h>
#include "parameters.h"
#include "counterparty2.h"
#include "state.h"
#include "data_reader2.h"

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


    Data_reader data;
    std::vector<Fx> fx_vector_temp;
    for (int k=0; k<params.fx_num/params.deals_at_once; ++k){
        // Get fx deal data
        fx_vector_temp.clear();
        data.get_next_data_fx(fx_vector_temp, params);
        thrust::device_vector<Fx> fx_vector(fx_vector_temp.begin(), fx_vector_temp.end());
        thrust::device_vector<float> cva_vector(fx_vector.size());
        thrust::transform(fx_vector.begin(), fx_vector.end(), cva_vector.begin(), calculate_cva_fx(params, num_of_steps));
        thrust::host_vector<float> cva_vector_host(cva_vector);

//        for (unsigned int i=0; i<cva_vector_host.size(); ++i){
//            std::cout << "cva " << k*params.deals_at_once+i+1 << " " << cva_vector_host[i] << " " << fx_vector_temp[i].fx_id << "\n";
//        }
    }

    end_time = clock() - program_start_time;
    std::cout << "Timing0 " << float(end_time)/CLOCKS_PER_SEC << " seconds since start.\n";

    std::vector<Swap> swap_vector_temp;
    for (int k=0; k<params.swap_num/params.deals_at_once; ++k){
        end_time = clock() - program_start_time;
        std::cout << "Timing1 " << k << " " << float(end_time)/CLOCKS_PER_SEC << " seconds since start.\n";
        // Get swap deal data
        swap_vector_temp.clear();
        end_time = clock() - program_start_time;
        std::cout << "Timing2 " << k << " " << float(end_time)/CLOCKS_PER_SEC << " seconds since start.\n";
        data.get_next_data_swap(swap_vector_temp, params);
        end_time = clock() - program_start_time;
        std::cout << "Timing3 " << k << " " << float(end_time)/CLOCKS_PER_SEC << " seconds since start.\n";
        thrust::device_vector<Swap> swap_vector(swap_vector_temp.begin(), swap_vector_temp.end());
        end_time = clock() - program_start_time;
        std::cout << "Timing4 " << k << " " << float(end_time)/CLOCKS_PER_SEC << " seconds since start.\n";
        thrust::device_vector<float> cva_vector(swap_vector.size());
        end_time = clock() - program_start_time;
        std::cout << "Timing5 " << k << " " << float(end_time)/CLOCKS_PER_SEC << " seconds since start.\n";
        thrust::transform(swap_vector.begin(), swap_vector.end(), cva_vector.begin(), calculate_cva_swap(params, num_of_steps));
        cudaDeviceSynchronize();
        end_time = clock() - program_start_time;
        std::cout << "Timing6 " << k << " " << float(end_time)/CLOCKS_PER_SEC << " seconds since start.\n";
        thrust::host_vector<float> cva_vector_host(cva_vector);
        end_time = clock() - program_start_time;
        std::cout << "Timing7 " << k << " " << float(end_time)/CLOCKS_PER_SEC << " seconds since start.\n";

//        for (unsigned int i=0; i<cva_vector_host.size(); ++i){
//            std::cout << "cva " << k*params.deals_at_once+i+1 << " " << cva_vector_host[i] << " " << swap_vector_temp[i].swap_id << "\n";
//        }
    }

    //multiply by recovery

    data.close_files();
    end_time = clock() - program_start_time;
    std::cout << "Timing: whole program " << float(end_time)/CLOCKS_PER_SEC << " seconds.\n";
    std::cout << "\n";
    return 0;
}

