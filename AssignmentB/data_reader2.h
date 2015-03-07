#ifndef DATA_READER_INCLUDED
#define DATA_READER_INCLUDED

#include <fstream>
#include <vector>

struct Data_reader{

    int fx_start_location, swap_start_location;
    int hazard_buckets[5];
    std::ifstream fx_details_infile, swap_details_infile;

    Data_reader();
    void get_next_data_fx(std::vector<Fx> &fx_vector, Parameters &params);
    void get_next_data_swap(std::vector<Swap> &swap_vector, Parameters &params);
    void close_files();
};

Data_reader::Data_reader()
{
    fx_start_location = 0;
    swap_start_location = 0;

    const char* hazard_buckets_filename="hazard_buckets.dat";
    const char* fx_details_filename="fx_details.dat";
    const char* swap_details_filename="swap_details.dat";

    // Get the list of hazard rate bucket endpoints
    std::ifstream hazard_buckets_infile;
    hazard_buckets_infile.open(hazard_buckets_filename);
    if (!hazard_buckets_infile.is_open()){
        std::cout << "ERROR: hazard_buckets.dat file could not be opened. Make sure to generate the bank data first. Exiting.\n";
        exit(1);
    }
    for (int i=0; i<5; ++i) hazard_buckets_infile >> hazard_buckets[i];
    hazard_buckets_infile.close();

    // Open the deal details
    fx_details_infile.open(fx_details_filename);
    if (!fx_details_infile.is_open()){
        std::cout << "ERROR: fx_details.dat file could not be opened. Exiting.\n";
        exit(1);
    }
    swap_details_infile.open(swap_details_filename);
    if (!swap_details_infile.is_open()){
        std::cout << "ERROR: swap_details.dat file could not be opened. Exiting.\n";
        exit(1);
    }
}

void Data_reader::get_next_data_fx(std::vector<Fx> &fx_vector, Parameters &params)
{
    // Read deals into memory
    int deals_handled=0;

    int fx_id, notional;
    float hazard_rate;
    char position;

    fx_details_infile.seekg(fx_start_location,fx_details_infile.beg);

    while (deals_handled < params.deals_at_once){
        fx_details_infile >> fx_id;
        fx_details_infile >> notional;
        fx_details_infile >> position;
        fx_details_infile >> hazard_rate;
        fx_vector.push_back(Fx(fx_id, notional, position, hazard_rate));
        ++deals_handled;
    }
    fx_start_location = fx_details_infile.tellg();
}

void Data_reader::get_next_data_swap(std::vector<Swap> &swap_vector, Parameters &params)
{
    // Read deals into memory
    int deals_handled=0;
    //clock_t program_start_time, end_time;
    //program_start_time = clock();

    int swap_id, notional, tenor;
    char position, denomination;
    float fixed_rate;
    float hazard_rate;

    swap_details_infile.seekg(swap_start_location,swap_details_infile.beg);

    while (deals_handled < params.deals_at_once){
        swap_details_infile >> swap_id;
        swap_details_infile >> denomination;
        swap_details_infile >> notional;
        swap_details_infile >> fixed_rate;
        swap_details_infile >> tenor;
        swap_details_infile >> position;
        swap_details_infile >> hazard_rate;
        swap_vector.push_back(Swap(swap_id, denomination, notional, fixed_rate, tenor, position, hazard_rate));
        ++deals_handled;
    }
    swap_start_location = swap_details_infile.tellg();
}

void Data_reader::close_files()
{
    fx_details_infile.close();
    swap_details_infile.close();
}

#endif // DATA_READER_INCLUDED