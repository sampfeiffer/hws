#ifndef LOGGING_INCLUDED
#define LOGGING_INCLUDED

#include <fstream>
#include <sstream>

struct Logging
{
    std::string log_filename;

    Logging(std::string filename);
    void write(std::stringstream &log_text);

};

Logging::Logging(std::string filename)
{
    log_filename = filename;

    // Empty (or create) file.
    std::ofstream log_file(log_filename);
    log_file.close();
}

void Logging::write(std::stringstream &log_text)
{
    std::ofstream log_file(log_filename, std::ios_base::out | std::ios_base::app);
    log_file << log_text.str();
    log_file.flush();
    log_text.str(""); // Empty text
}

#endif // LOGGING_INCLUDED
