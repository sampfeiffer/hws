#ifndef FUNCTORS_INCLUDED
#define FUNCTORS_INCLUDED

#include <thrust/functional.h>

// convert a linear index to a row index
template <typename T>
struct linear_index_to_row_index : public thrust::unary_function<T,T>
{
    T num_of_deals; // number of rows

    __host__ __device__
    linear_index_to_row_index(T num_of_deals_) : num_of_deals(num_of_deals_) {}
    __host__ __device__
    T operator()(T i)
    {
        return i % num_of_deals;
    }
};

// divide by a number
template <typename T>
struct divide_by
{
    T divisor; // number to divide by

    __host__ __device__
    divide_by(T divisor_) : divisor(divisor_) {}
    __host__ __device__
    T operator()(T i)
    {
        return i / divisor;
    }
};

//// Calculate the cva of an FX deal over a specified number of paths.
//struct calculate_cva_fx{
//    int num_of_steps, paths_per_gpu;
//    State* path_ptr;
//
//    calculate_cva_fx(int num_of_steps_, int paths_per_gpu_, State* path_ptr_) : num_of_steps(num_of_steps_), paths_per_gpu(paths_per_gpu_), path_ptr(path_ptr_)
//    {}
//    __device__ __host__
//    float operator()(Fx &fx) {
//        float cva, sum=0;
//        float prob_default;
//        State world_state;
//        for (int path=0; path<paths_per_gpu; ++path){
//            cva=0;
//            for (int i=0; i<num_of_steps; ++i){
//                world_state = path_ptr[path*num_of_steps+i];
//                prob_default = std::exp(-fx.hazard_rate*(world_state.time-1)/float(360.0)) - std::exp(-fx.hazard_rate*world_state.time/float(360.0));
//                cva += world_state.cva_disc_factor * prob_default * max(fx.value(world_state.fx_rate),float(0.0));
//            }
//            sum += cva;
//        }
//        return sum/paths_per_gpu;
//    }
//};

// Calculate the cva of an swap over a specified number of paths.
struct calculate_cva{
    int num_of_steps, paths_per_gpu;
    State* path_ptr;

    calculate_cva(int num_of_steps_, int paths_per_gpu_, State* path_ptr_) : num_of_steps(num_of_steps_), paths_per_gpu(paths_per_gpu_), path_ptr(path_ptr_)
    {}

    __device__ __host__
    float operator()(Fx &fx) {
        float cva, sum=0;
        float prob_default;
        State world_state;
        for (int path=0; path<paths_per_gpu; ++path){
            cva=0;
            for (int i=0; i<num_of_steps; ++i){
                world_state = path_ptr[path*num_of_steps+i];
                prob_default = std::exp(-fx.hazard_rate*(world_state.time-1)/float(360.0)) - std::exp(-fx.hazard_rate*world_state.time/float(360.0));
                cva += world_state.cva_disc_factor * prob_default * max(fx.value(world_state.fx_rate),float(0.0));
            }
            sum += cva;
        }
        return sum/paths_per_gpu;
    }

    __device__ __host__
    float operator()(Swap &sw) {
        float cva, sum=0;
        float prob_default;
        State world_state;
        for (int path=0; path<paths_per_gpu; ++path){
            cva=0;
            for (int i=0; i<num_of_steps; ++i){
                world_state = path_ptr[path*num_of_steps+i];
                prob_default = std::exp(-sw.hazard_rate*(world_state.time-1)/float(360.0)) - std::exp(-sw.hazard_rate*world_state.time/float(360.0));
                cva += world_state.cva_disc_factor * prob_default * max(sw.value(world_state),float(0.0));
            }
            sum += cva;
        }
        return sum/paths_per_gpu;
    }
};

#endif // FUNCTORS_INCLUDED
