#ifndef NELSON_SIEGEL_INCLUDED
#define NELSON_SIEGEL_INCLUDED

#include <thrust/random/linear_congruential_engine.h>
#include <thrust/random/normal_distribution.h>

//#include <random>
//std::default_random_engine generator;
//std::normal_distribution<double> standard_normal;
//thrust::minstd_rand generator;
//thrust::random::normal_distribution<float> standard_normal(0.0f, 1.0f);

struct NelsonSiegel{
    float step_size;
    float betas[4], betas_beg[4];
    float alphas[4], sigmas[4];

    __device__ __host__ NelsonSiegel(float step_size_, float betas_beg_[], float alphas_[], float sigmas_[]);
    __device__ __host__ double yield(float t);
    __device__ __host__ void sim_next_step();
};

__device__ __host__
NelsonSiegel::NelsonSiegel(float step_size_, float betas_beg_[], float alphas_[], float sigmas_[])
{
    step_size = step_size_;
    int i;
    for (i=0; i<4; ++i){
        betas_beg[i] = betas_beg_[i];
        betas[i] = betas_beg[i];
    }
    for (i=0; i<4; ++i) alphas[i] = alphas_[i];
    for (i=0; i<4; ++i) sigmas[i] = sigmas_[i];
}

__device__ __host__
double NelsonSiegel::yield(float t)
{
    if (t==0) return 0;
    return betas[0] + betas[1]*(1-exp(-t/betas[3]))/(t/betas[3]) + betas[2]*((1-exp(-t/betas[3]))/(t/betas[3])-exp(-t/betas[3]));
}

__device__ __host__
void NelsonSiegel::sim_next_step()
{
    thrust::minstd_rand generator;
    thrust::random::normal_distribution<float> standard_normal(0.0f, 1.0f);
    for (int i=0; i<4; ++i){
        betas[i] = betas_beg[i] + alphas[i]*(betas_beg[i]-betas[i]) + sigmas[i]*sqrt(step_size/365)*standard_normal(generator);
    }
}

#endif // NELSON_SIEGEL_INCLUDED
