#ifndef FX_INCLUDED
#define FX_INCLUDED

#include <iostream>
#include <cmath>

struct Fx{
    int fx_id, notional;
    char position;
    float hazard_rate;

    __device__ __host__ Fx(){}
    __device__ __host__ Fx(int fx_id_, int notional_, char position_, float hazard_rate_); //Constructor
    __device__ __host__ float value(float fx_rate_cur);

};

// Constructor
__device__ __host__
Fx::Fx(int fx_id_, int notional_, char position_, float hazard_rate_)
{
    fx_id = fx_id_;
    notional = notional_;
    position = position_;
    hazard_rate = hazard_rate_;
}

// Calculate the value of the FX under the current FX rate
__device__ __host__
float Fx::value(float fx_rate_cur)
{
    int sign=1;
    if (position == 's') sign=-1;

    return sign*-1*notional*max(fx_rate_cur,float(0.0));
}

#endif // FX_INCLUDED
