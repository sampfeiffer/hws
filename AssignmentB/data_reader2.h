#ifndef DATA_READER_INCLUDED
#define DATA_READER_INCLUDED

#include <fstream>
#include <vector>

struct Data_reader{

    int cp_id, bucket, start_location;
    int hazard_buckets[5];
    float hazard_rate;
    std::ifstream counterparty_deals_infile, fx_details_infile, swap_details_infile;

    Data_reader();
    void get_next_data(std::vector<Counterparty> &cp_vector, Parameters &params);
    void close_files();
};

Data_reader::Data_reader()
{
    cp_id = 1;
    bucket = 0;
    start_location = 2;
    hazard_rate=0.10;

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


void Data_reader::get_next_data(std::vector<Counterparty> &cp_vector, Parameters &params)
{
    // Read deals into memory
    int current_id=1, deal_id, deals_handled=0, bucket=0;

    int fx_id, swap_id, notional, tenor, start_of_data, fx_count, swap_count;
    char position, denomination;
    float fixed_rate;

    counterparty_deals_infile.seekg(start_location,counterparty_deals_infile.beg);

    while (deals_handled <= params.deals_at_once){
        if (cp_id > hazard_buckets[bucket]){
            ++bucket;
            hazard_rate -= 0.02;
        }
        start_of_data = counterparty_deals_infile.tellg();
        fx_count = 0;
        swap_count = 0;

        do{
            counterparty_deals_infile >> deal_id;
            if (deal_id<params.fx_num) ++fx_count;
            else ++swap_count;
            counterparty_deals_infile >> current_id;
        } while(current_id == cp_id);
        start_location = counterparty_deals_infile.tellg();
        counterparty_deals_infile.seekg(start_of_data,counterparty_deals_infile.beg);

        Counterparty cp(cp_id, hazard_rate, fx_count, swap_count);
        do{
            counterparty_deals_infile >> deal_id;
            if (deal_id<params.fx_num){
                fx_details_infile >> fx_id;
                fx_details_infile >> notional;
                fx_details_infile >> position;
                cp.add_fx(fx_id, notional, position);
            }
            else {
                swap_details_infile >> swap_id;
                swap_details_infile >> denomination;
                swap_details_infile >> notional;
                swap_details_infile >> fixed_rate;
                swap_details_infile >> tenor;
                swap_details_infile >> position;
                cp.add_swap(swap_id, denomination, notional, fixed_rate, tenor, position);
            }
            ++deals_handled;
            counterparty_deals_infile >> current_id;
        } while(current_id == cp_id);
        cp_vector.push_back(cp);
        ++cp_id;
    }
}

void Data_reader::close_files()
{
    fx_details_infile.close();
    swap_details_infile.close();
}

#endif // DATA_READER_INCLUDED
