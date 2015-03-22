#ifndef GET_DRIFT_INCLUDED
#define GET_DRIFT_INCLUDED

#include <iostream>
#include <fstream>
#include "parameters.h"

float get_drift(char* tick_data_filename, int &chars_per_line)
{
    std::ifstream tick_data_infile;
    tick_data_infile.open(tick_data_filename);
    if (!tick_data_infile.is_open()){
        std::cout << "ERROR: tick_data.dat file could not be opened. Exiting.\n";
        exit(1);
    }

    float start_data, end_data;
    tick_data_infile >> start_data;
    tick_data_infile.seekg(-(chars_per_line+1), tick_data_infile.end);
    tick_data_infile >> end_data;

    tick_data_infile.close();
    return (end_data-start_data)/100;
}


#endif // GET_DRIFT_INCLUDED
