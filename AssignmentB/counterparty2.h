#ifndef COUNTERPARTY_INCLUDED
#define COUNTERPARTY_INCLUDED

#include <cmath>
#include "fx2.h"
#include "swap2.h"

struct Counterparty{
    int cp_id, num_of_fx, num_of_swap;

    Counterparty(){};
    Counterparty(int cp_id_, int fx_count, int swap_count); //Constructor
};

// Constructor
Counterparty::Counterparty(int cp_id_, int fx_count, int swap_count)
{
    cp_id = cp_id_;
    num_of_fx = fx_count;
    num_of_swap = swap_count;
}

#endif // COUNTERPARTY_INCLUDED
