#ifndef LOGGING_INCLUDED
#define LOGGING_INCLUDED

#include <ctime>
#include <chrono>
#include <sys/sysinfo.h>
#include "parameters.h"

// Handles the program logging
struct Logging
{
    int line_id;
    std::string log_filename;
    std::ofstream log_file;

    Logging(std::string filename); //Constructor
    ~Logging(); //Destructor
    void start_info(Parameters &params, int argc, char* argv[], int &size);
    void hardware_info();
    void write(std::stringstream &log_text);

};

// Constructor
Logging::Logging(std::string filename)
{
    line_id = 0;
    log_filename = filename;

    // Empty (or create) file.
    log_file.open(log_filename);
    log_file.close();
    log_file.open(log_filename, std::ios_base::out | std::ios_base::app);
}

//Destructor
Logging::~Logging()
{
    log_file.close();
}

// Program starting info
void Logging::start_info(Parameters &params, int argc, char* argv[], int &size)
{
    log_file << "===========================================\n"
             << "INITIAL INFORMATION\n"
             << "===========================================\n";

    std::chrono::system_clock::time_point today = std::chrono::system_clock::now();
    time_t tt;
    tt = std::chrono::system_clock::to_time_t ( today );
    log_file << "Current datetime: " << ctime(&tt);

    log_file << "Program name: " << argv[0] << "\n";
    for (int i=1; i<argc; ++i){
        log_file << "run-time parameter " << i << ": " << argv[i] << "\n";
    }
    log_file << params.print();

    log_file << "Number of threads: " << size << "\n";
    hardware_info();

    log_file << "===========================================\n\n";
}

void Logging::hardware_info()
{
    struct sysinfo info;

    if (sysinfo(&info) != 0)
        log_file << "sysinfo: error reading system statistics\n";
    else {
        log_file << "Total RAM: " << info.totalram/(1024*1024) << "MB\n"
                 << "Free RAM: " << info.freeram/(1024*1024) << "MB\n";
    }
}

// Flush the log_text to the log file.
void Logging::write(std::stringstream &log_text)
{
    log_file << log_text.str();
    log_file.flush();
    log_text.str(""); // Empty log_text
}

#endif // LOGGING_INCLUDED
