#ifndef STATE_INCLUDED
#define STATE_INCLUDED

#include "nelson_siegel.h"

struct State{
    float step_size;
    double fx_rate_beg, fx_rate, fx_vol;
    NelsonSiegel amer, euro;

    State(Parameters &params);
    void sim_next_step();
};

State::State(Parameters &params):
    amer(params.step_size, params.amer_betas, params.amer_alphas, params.amer_sigmas),
    euro(params.step_size, params.euro_betas, params.euro_alphas, params.euro_sigmas)
{
    step_size = params.step_size;
    fx_rate_beg = params.eur_usd_rate;
    fx_vol = params.eur_usd_vol;
    fx_rate = fx_rate_beg;
}

void State::sim_next_step()
{
    amer.sim_next_step();
    euro.sim_next_step();
    std::normal_distribution<double> normal_fx(0.0,1.0);
    fx_rate += fx_vol*sqrt(step_size/365)*normal_fx(generator);
}


#endif // STATE_INCLUDED
