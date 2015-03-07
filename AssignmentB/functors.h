#ifndef FUNCTORS_INCLUDED
#define FUNCTORS_INCLUDED

#include <thrust/functional.h>

// convert a linear index to a row index
template <typename T>
struct linear_index_to_row_index : public thrust::unary_function<T,T>
{
    T C; // number of columns
    __host__ __device__
    linear_index_to_row_index(T C_) : C(C_) {}
    __host__ __device__
    T operator()(T i)
    {
    return i / C;
    }
};

struct calculate_cva_fx{
    Parameters params;
    int num_of_steps, paths_per_gpu;

    calculate_cva_fx(Parameters params_, int num_of_steps_, int paths_per_gpu_) : params(params_), num_of_steps(num_of_steps_), paths_per_gpu(paths_per_gpu_)
    {}
    __device__ __host__
    float operator()(Fx &fx) {
        float cva, sum=0;
        float prob_default;
        for (int path=0; path<paths_per_gpu; ++path){
            cva=0;
            State world_state(params);
            for (int i=0; i<num_of_steps; ++i){
                world_state.sim_next_step();
                prob_default = std::exp(-fx.hazard_rate*(world_state.time-1)/float(360.0)) - std::exp(-fx.hazard_rate*world_state.time/float(360.0));
                cva += world_state.cva_disc_factor * prob_default * max(fx.value(world_state.fx_rate),float(0.0));
            }
            sum += cva;
        }
        return sum/paths_per_gpu;
    }
};

struct calculate_cva_swap{
    Parameters params;
    int num_of_steps, paths_per_gpu;

    calculate_cva_swap(Parameters params_, int num_of_steps_, int paths_per_gpu_) : params(params_), num_of_steps(num_of_steps_), paths_per_gpu(paths_per_gpu_)
    {}
    __device__ __host__
    float operator()(Swap &sw) {
        float cva, sum=0;
        float prob_default;
        for (int path=0; path<paths_per_gpu; ++path){
            cva=0;
            State world_state(params);
            for (int i=0; i<num_of_steps; ++i){
                world_state.sim_next_step();
                prob_default = std::exp(-sw.hazard_rate*(world_state.time-1)/float(360.0)) - std::exp(-sw.hazard_rate*world_state.time/float(360.0));
                cva += world_state.cva_disc_factor * prob_default * max(sw.value(world_state),float(0.0));
            }
            sum += cva;
        }
        return sum/paths_per_gpu;
    }
};

#endif // FUNCTORS_INCLUDED