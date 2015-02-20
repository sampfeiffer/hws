#ifndef LOGGING_INCLUDED
#define LOGGING_INCLUDED

#include <fstream>
#include <sstream>
#include "usage.h"

// Handles the program logging
struct Logging
{
    std::string log_filename;

    Logging(std::string filename); //Constructor
    void write(std::stringstream &log_text, bool usage=false);

};

// Constructor
Logging::Logging(std::string filename)
{
    log_filename = filename;

    // Empty (or create) file.
    std::ofstream log_file(log_filename);
    log_file.close();
}

// Flush the log_text to the log file.
void Logging::write(std::stringstream &log_text, bool usage)
{
    std::ofstream log_file(log_filename, std::ios_base::out | std::ios_base::app);
    if (usage){
        log_text << get_info();
    }
    log_file << log_text.str();
    log_file.flush();
    log_text.str(std::string()); // Empty log_text
}

#endif // LOGGING_INCLUDED
