#ifndef FX_INCLUDED
#define FX_INCLUDED

#include <iostream>
#include <cmath>

struct Fx{
    int fx_id, notional;
    char position;

    __device__ __host__ Fx(int fx_id_, int notional_, char position_); //Constructor
    //void print();
    //void print_short();
    __device__ __host__ double value(double fx_rate_cur);

};

// Constructor
__device__ __host__
Fx::Fx(int fx_id_, int notional_, char position_)
{
    fx_id = fx_id_;
    notional = notional_;
    position = position_;
}

//void Fx::print()
//{
//    std::cout << "\nfx_id " << fx_id
//              << "\nnotional " << notional
//              << "\nposition " << position << "\n";
//}
//
//void Fx::print_short()
//{
//    std::cout << "    " << fx_id << " " << notional << " " << position << "\n";
//}

__device__ __host__
double Fx::value(double fx_rate_cur)
{
    int sign=1;
    if (position == 's') sign=-1;

    return sign*-1*notional*max(fx_rate_cur,0.0);
}

#endif // FX_INCLUDED
