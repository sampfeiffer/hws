#ifndef SWAP_INCLUDED
#define SWAP_INCLUDED

#include <iostream>
#include "state.h"

struct Swap{
    int swap_id, notional, tenor;
    char denomination, position;
    float fixed_rate, hazard_rate;

    __device__ __host__ Swap(int swap_id_, char denomination_, int notional_, float fixed_rate_, int tenor_, int position_, float hazard_rate_); //Constructor
    //void print();
    //void print_short();
    __device__ __host__ float value(State &world_state);
};

// Constructor
__device__ __host__
Swap::Swap(int swap_id_, char denomination_, int notional_, float fixed_rate_, int tenor_, int position_, float hazard_rate_)
{
    swap_id = swap_id_;
    denomination = denomination_;
    notional = notional_;
    fixed_rate = fixed_rate_;
    tenor = tenor_;
    position = position_;
    hazard_rate = hazard_rate_;
}

//void Swap::print()
//{
//    std::cout << "\nswap_id " << swap_id
//              << "\ndenomination " << denomination
//              << "\nnotional " << notional
//              << "\nfixed_rate " << fixed_rate
//              << "\ntenor " << tenor
//              << "\nposition " << position << "\n";
//}
//
//void Swap::print_short()
//{
//    std::cout << "    " << swap_id << " " << denomination << " " << notional << " " << fixed_rate << " "  << tenor << " " << position << "\n";
//}

__device__ __host__
float Swap::value(State &world_state)
{
    float fixed_leg, float_leg, coupon_date=tenor-world_state.time/float(360.0), next_reset = (30-(world_state.time%30))/float(360.0);

    if (coupon_date < 0) return 0;

    if (denomination == 'a'){
        //Fixed leg
        fixed_leg = float(1.0)/(1+world_state.amer.yield(coupon_date)*coupon_date); //bullet
        while (coupon_date > 0){
            fixed_leg += float(1.0)/(1+world_state.amer.yield(coupon_date)*coupon_date) * fixed_rate/float(12.0); //coupons
            coupon_date -= float(1.0)/12;
        }

        //Floating leg
        float_leg = float(1.0)/(1+world_state.amer.yield(next_reset)*next_reset);

        int sign=1;
        if (position == 's') sign=-1;

        return sign*(float_leg-fixed_leg)*notional;
    }
    else{
        //Fixed leg
        fixed_leg = float(1.0)/(1+world_state.euro.yield(coupon_date)*coupon_date); //bullet
        while (coupon_date > 0){
            fixed_leg += float(1.0)/(1+world_state.euro.yield(coupon_date)*coupon_date) * fixed_rate/float(12.0); //coupons
            coupon_date -= float(1.0)/12;
        }

        //Floating leg
        float_leg = float(1.0)/(1+world_state.euro.yield(next_reset)*next_reset);

        int sign=1;
        if (position == 's') sign=-1;

        return sign*(float_leg-fixed_leg)*notional*world_state.fx_rate;
    }
}

#endif // SWAP_INCLUDED
