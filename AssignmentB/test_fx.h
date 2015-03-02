#ifndef FX_INCLUDED
#define FX_INCLUDED

#include <iostream>

struct Fx{
    int fx_id, notional;
    char position;

    Fx(int fx_id_, int notional_, char position_); //Constructor
    void print();
    void print_short();
    double value(double fx_rate_beg, double fx_rate_cur);

};

// Constructor
Fx::Fx(int fx_id_, int notional_, char position_)
{
    fx_id = fx_id_;
    notional = notional_;
    position = position_;
}

void Fx::print()
{
//    std::cout << "\nfx_id " << fx_id
//              << "\nnotional " << notional
//              << "\nposition " << position << "\n";
    printf("\nfx_id %i", fx_id);
    printf("\nnotional %i", notional);
    printf("\nposition %c \n", position);
}

void Fx::print_short()
{
    //std::cout << "    " << fx_id << " " << notional << " " << position << "\n";
    printf("    %i %i %c", fx_id, notional, position);
}

double Fx::value(double fx_rate_beg, double fx_rate_cur)
{
    int sign=1;
    if (position == 's') sign=-1;

    return sign*notional*(std::max(fx_rate_cur,0.0)/fx_rate_beg - 1);
}

#endif // FX_INCLUDED
