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

    const char* counterparty_deals_filename="counterparty_deals.dat";
    std::ifstream counterparty_deals_infile;
    counterparty_deals_infile.open(counterparty_deals_filename);
    if (!counterparty_deals_infile.is_open()){
        std::cout << "ERROR: counterparty_deals.dat file could not be opened. Exiting.\n";
        exit(1);
    }



    Data_reader data;
    std::vector<Fx> fx_vector_temp;
    thrust::host_vector<float> cva_vector_host(params.fx_num);
    for (int k=0; k<params.fx_num/params.deals_at_once; ++k){
        // Get fx deal data
        fx_vector_temp.clear();
        data.get_next_data_fx(fx_vector_temp, params);
        thrust::device_vector<Fx> fx_vector(fx_vector_temp.begin(), fx_vector_temp.end());
        thrust::device_vector<float> cva_vector(fx_vector.size());
        thrust::transform(fx_vector.begin(), fx_vector.end(), cva_vector.begin(), calculate_cva_fx(params, num_of_steps));
        thrust::copy(cva_vector.begin(), cva_vector.end(), cva_vector_host.begin()+k*params.deals_at_once);

//        for (unsigned int i=0; i<cva_vector_host.size(); ++i){
//            std::cout << "cva " << k*params.deals_at_once+i+1 << " " << cva_vector_host[i] << " " << fx_vector_temp[i].fx_id << "\n";
//        }
    }
    std::cout << "size " << cva_vector_host.size() << "\n";

    int total_deals = params.fx_num + params.swap_num;
    int cp_id=1, cp_id_read, deal_id_read;
    float cva_temp=0;
    std::vector<float> total_cva;
    int temp_count=0;

    counterparty_deals_infile >> cp_id_read;
    for (int i=0; i<total_deals; ++i){
        counterparty_deals_infile >> deal_id_read;
        if (deal_id_read < params.fx_num){
            cva_temp += cva_vector_host[deal_id_read-1];
            ++temp_count;
        }
        counterparty_deals_infile >> cp_id_read;
        if (cp_id_read > cp_id){
            total_cva.push_back(cva_temp);
            cva_temp = 0;
            std::cout << "cva cps " << cp_id << " " << total_cva[cp_id-1] << " " << temp_count << "\n";
            ++cp_id;
            temp_count=0;
        }
    }

    for (unsigned int i=0; i<50; ++i){
        std::cout << "cva deals " << i << " " << cva_vector_host[i] << "\n";
    }

    std::vector<Swap> swap_vector_temp;
    for (int k=0; k<params.swap_num/params.deals_at_once; ++k){
        // Get swap deal data
        swap_vector_temp.clear();
        data.get_next_data_swap(swap_vector_temp, params);
        thrust::device_vector<Swap> swap_vector(swap_vector_temp.begin(), swap_vector_temp.end());
        thrust::device_vector<float> cva_vector(swap_vector.size());
        thrust::transform(swap_vector.begin(), swap_vector.end(), cva_vector.begin(), calculate_cva_swap(params, num_of_steps));
        thrust::copy(cva_vector.begin(), cva_vector.end(), cva_vector_host.begin()+k*params.deals_at_once);

//        for (unsigned int i=0; i<cva_vector_host.size(); ++i){
//            std::cout << "cva " << k*params.deals_at_once+i+1 << " " << cva_vector_host[i] << " " << swap_vector_temp[i].swap_id << "\n";
//        }
    }

    std::cout << "size " << cva_vector_host.size() << "\n";
    std::cout << "cva " << thrust::reduce(cva_vector_host.begin(), cva_vector_host.end()) << "\n";

    //multiply by recovery

    data.close_files();

    counterparty_deals_infile.close();

    end_time = clock() - program_start_time;
    std::cout << "Timing: whole program " << float(end_time)/CLOCKS_PER_SEC << " seconds.\n";
    std::cout << "\n";
    return 0;
}

