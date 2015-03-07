#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <vector>
#include <time.h>
#include "parameters.h"
#include "counterparty.h"
#include "state.h"
#include "data_reader.h"

struct calculate_cva{
    Parameters params;
    int num_of_steps;

    calculate_cva(Parameters params_, int num_of_steps_) : params(params_), num_of_steps(num_of_steps_)
    {}
    __device__ __host__
    float operator()(Counterparty &cp) {
        float cva=0;
        float total_value;
        State world_state(params);
        for (int i=0; i<num_of_steps; ++i){
            total_value = 0;
            world_state.sim_next_step();
            // CVA for fx
            for (unsigned int fx=0; fx<cp.num_of_fx; ++fx){
                total_value += max(cp.fx_deals[fx]->value(world_state.fx_rate),float(0.0));
            }
            // CVA for swaps
            for (unsigned int sw=0; sw<cp.num_of_swap; ++sw){
//                total_value += max(cp.swap_deals[sw]->value(world_state),float(0.0));
            }
            cva += world_state.cva_disc_factor * cp.prob_default(world_state.time) * total_value;
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
    //params.print();

    // Get counterparty data and store info in cp_vector_temp
    thrust::device_vector<Counterparty> cp_vector_temp;
    Data_reader data;
    data.get_next_data(cp_vector_temp, params);
    //std::cout << "test info " << cp_vector_temp[1].fx_deals[0]->fx_id << "\n";

    thrust::device_vector<Counterparty> cp_vector(cp_vector_temp.begin(), cp_vector_temp.end());

    int num_of_steps = params.days_in_year*params.time_horizon/params.step_size;

    end_time = clock() - program_start_time;
    std::cout << "Timing: whole program " << float(end_time)/CLOCKS_PER_SEC << " seconds.\n";

    //thrust::host_vector<float> cva_vector(cp_vector.size());
    thrust::device_vector<float> cva_vector(cp_vector.size());
    std::cout << "here1 " << cva_vector.size() << "\n";
    thrust::transform(cp_vector.begin(), cp_vector.end(), cva_vector.begin(), calculate_cva(params, num_of_steps));
    std::cout << "here2\n";
    std::cout << "size1 " << cva_vector.size() << "\n";
    thrust::host_vector<float> cva_vector_host(cva_vector);
    std::cout << "size2 " << cva_vector_host.size() << "\n";

    for (unsigned int i=0; i<cva_vector_host.size(); ++i){
        std::cout << "here in " << i+1 << "\n";
        std::cout << "cva " << i+1 << " " << cva_vector_host[i] << "\n";
    }

    data.close_files();
    std::cout << "\n";
    return 0;
}

