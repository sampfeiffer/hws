#ifndef STATE_INCLUDED
#define STATE_INCLUDED

#include "nelson_siegel.h"
#include <cmath>

struct State{
    float step_size, cva_disc_rate;
    double fx_rate_beg, fx_rate, fx_vol, cva_disc_factor;
    int time;
    NelsonSiegel amer, euro;

    __device__ __host__ State(Parameters &params);
    __device__ __host__ void sim_next_step();
};

__device__ __host__
State::State(Parameters &params):
    amer(params.step_size, params.amer_betas, params.amer_alphas, params.amer_sigmas),
    euro(params.step_size, params.euro_betas, params.euro_alphas, params.euro_sigmas)
{
    step_size = params.step_size;
    fx_rate_beg = params.eur_usd_rate;
    fx_vol = params.eur_usd_vol;
    fx_rate = fx_rate_beg;
    time = 0;
    cva_disc_rate = params.cva_disc_rate;
    cva_disc_factor = pow(double(1+cva_disc_rate), -time/360.0);
}

__device__ __host__
void State::sim_next_step()
{
    thrust::minstd_rand generator1;
    thrust::random::normal_distribution<float> standard_normal1(0.0f, 1.0f);
    amer.sim_next_step();
    euro.sim_next_step();
    fx_rate += fx_vol*sqrt(step_size/360.0)*standard_normal1(generator1);
    ++time;
    cva_disc_factor = pow(double(1+cva_disc_rate), -time/360.0);
}


#endif // STATE_INCLUDED
