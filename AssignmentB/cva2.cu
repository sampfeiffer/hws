#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <vector>
#include <time.h>
#include "parameters.h"
#include "counterparty2.h"
#include "state.h"
#include "data_reader2.h"

struct calculate_cva{
    Parameters params;
    int num_of_steps;

    calculate_cva(Parameters params_, int num_of_steps_) : params(params_), num_of_steps(num_of_steps_)
    {}
    __device__ __host__
    float operator()(Fx &fx) {
        float cva=0;
        float prob_default;
        State world_state(params);
        for (int i=0; i<num_of_steps; ++i){
            world_state.sim_next_step();
            prob_default = std::exp(-fx.hazard_rate*(world_state.time-1)/float(360.0)) - std::exp(-fx.hazard_rate*world_state.time/float(360.0));
            cva += world_state.cva_disc_factor * prob_default * max(fx.value(world_state.fx_rate),float(0.0));;
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
    std::vector<Fx> fx_vector_temp;
    Data_reader data;
    data.get_next_data_fx(fx_vector_temp, params);

    thrust::device_vector<Fx> fx_vector(fx_vector_temp.begin(), fx_vector_temp.end());

    int num_of_steps = params.days_in_year*params.time_horizon/params.step_size;

    end_time = clock() - program_start_time;
    std::cout << "Timing: whole program " << float(end_time)/CLOCKS_PER_SEC << " seconds.\n";

    thrust::device_vector<float> cva_vector(fx_vector.size());
    //std::cout << "here1 " << cva_vector.size() << "\n";
    thrust::transform(fx_vector.begin(), fx_vector.end(), cva_vector.begin(), calculate_cva(params, num_of_steps));
    //std::cout << "here2\n";
    thrust::host_vector<float> cva_vector_host(cva_vector);

    for (unsigned int i=0; i<cva_vector_host.size(); ++i){
        //std::cout << "here in " << i+1 << "\n";
        std::cout << "cva " << i+1 << " " << cva_vector_host[i] << "\n";
    }

    data.close_files();
    std::cout << "\n";
    return 0;
}

