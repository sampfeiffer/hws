#ifndef NELSON_SIEGEL_INCLUDED
#define NELSON_SIEGEL_INCLUDED

#include <random>

struct NelsonSiegel{
    float step_size;
    double betas[4], betas_beg[4];
    float alphas[4], sigmas[4];
    std::default_random_engine generator;
    std::normal_distribution<double> standard_normal;

    NelsonSiegel(float step_size_, double betas_beg_[], float alphas_[], float sigmas_[]);
    double yield(float t);
    void sim_next_step();
};

NelsonSiegel::NelsonSiegel(float step_size_, double betas_beg_[], float alphas_[], float sigmas_[])
{
    step_size = step_size_;
    int i;
    for (i=0; i<4; ++i){
        betas_beg[i] = betas_beg_[i];
        betas[i] = betas_beg[i];
    }
    for (i=0; i<4; ++i) alphas[i] = alphas_[i];
    for (i=0; i<4; ++i) sigmas[i] = sigmas_[i];
    //standard_normal(0.0,1.0);
}

double NelsonSiegel::yield(float t)
{
    return betas[0] + betas[1]*(1-exp(-t/betas[3]))/(t/betas[3]) + betas[2]*((1-exp(-t/betas[3]))/(t/betas[3])-exp(-t/betas[3]));
}

void NelsonSiegel::sim_next_step()
{
    for (int i=0; i<4; ++i){
        //betas[i] = betas_beg[i] + alphas[i]*(betas_beg[i]-betas[i]) + sigmas[i]*sqrt(step_size/365)*standard_normal(generator);
        betas[i] = betas[i] + alphas[i]*(betas_beg[i]-betas[i]) + sigmas[i]*sqrt(step_size/365)*standard_normal(generator);
    }
}

#endif // NELSON_SIEGEL_INCLUDED
