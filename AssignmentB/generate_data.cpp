#include "parameters.h"


int main(int argc, char *argv[])
{
    srand(time(NULL)); //Set the seed for the random number generator
    std::string parameters_filename = "parameters.txt";
    std::string bank_data_filename = "bank_data.txt";

    Parameters params("parameters.txt");

    std::ofstream bank_data;
    bank_data.open(bank_data_filename);
    if (!bank_data.is_open()){
        std::cout << "ERROR: " << bank_data_filename << " bank_data.txt file could not be opened. Exiting.\n";
        exit(1);
    }

    float hazard_rate;
    int cp_fx;
    int cp_swaps;

    for (int i=1; i<=params.counterparty_num; ++i){
        hazard_rate = 0.02 + (rand()%5)/50.0; //Uniformly random hazard rate
        if(rand()%2){
            cp_fx = 1;
            cp_swaps = 0;
            --params.fx_num;
        }
        else {
            cp_fx = 0;
            cp_swaps = 1;
            --params.swap_num;
        }

        std::cout << i << "," << hazard_rate << "," << cp_fx << "," << cp_swaps <<"\n";

    }

    params.print();


    bank_data.close();
    return 0;
}
