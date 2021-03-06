#include <cstdlib>
#include <thread>
#include <vector>
#include "scrub_tick.h"
#include "timing.h"
#include "scrub_functions.h"
#include "logging.h"

// Memory mapping headers.
#include <sys/mman.h>
#include <sys/types.h>
#include <fcntl.h>
#include <unistd.h>
#include <assert.h>

int main(int argc, char *argv[])
{
    // Input error checking
    if (argc < 3){
        std::cout << "ERROR: program needs 2 or 3 parameters\n"
                  << "Format is\n"
                  << "     ./scrub.out [input_data_filename] [num_of_threads] [copy_file]\n"
                  << "copy_file defaults to true if not specified. To not copy the file, put \"no\" as the third parameter.";
        return 1;
    }

    std::stringstream log_text;
    Logging logger("log_file_scrub.txt");

    const char *signal_filename = "signal.txt";
    bool copy_file_bool = false;

    // Copy data file into a file called "signal.txt". All operations take place on this file.
    // Ignore this if the user inputs "no" as the third parameter.
    Timing copy_file_time;
    copy_file_time.start_timing();
    if (!argv[3] || (argv[3] && strcmp(argv[3], "no") != 0)) copy_file_bool = true;
    if (copy_file_bool){
        copy_file(argv[1], signal_filename);
    }
    copy_file_time.end_timing();
    log_text << copy_file_time.print("file copy");
    logger.write(log_text);

    // Start timing program
    Timing program_time;
    program_time.start_timing();

    unsigned int filesize = getFilesize(signal_filename);
    int fd = open(signal_filename, O_RDWR, 0); //Open file
    assert(fd != -1); // Error check
    char* mapped = static_cast<char*>(mmap(NULL, filesize, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE, fd, 0)); //Execute mmap
    assert(mapped != NULL); // Error check

    // Threading and filesize info.
    unsigned int num_threads = atoi(argv[2]);
    unsigned int possible_threads = std::thread::hardware_concurrency();
    log_text << possible_threads << " concurrent threads are supported.\n";
    log_text << "Number of threads: " << num_threads << "\n";
    log_text << "Filesize: " << filesize << " characters\n\n";
    logger.write(log_text);

    // Split up file between threads
    std::vector<int> file_split(num_threads+1);
    for (int i=0; i<num_threads+1; ++i){
        file_split[i] = i*filesize/num_threads;
    }

    std::thread thread[num_threads];
    std::vector<std::stringstream> noise(num_threads);

    // Launch threads.
    for (int i=0; i<num_threads; ++i) {
        thread[i] = std::thread(process_data, mapped, file_split[i], file_split[i+1], &noise[i]);
        log_text << "Thread " << i << " launched\n";
        logger.write(log_text, true);
    }

    // Join threads to the main thread of execution.
    for (int i=0; i<num_threads; ++i){
        thread[i].join();
    }
    log_text << "Back in main thread.\n";
    logger.write(log_text, true);

    // Write the noise to noise.txt
    Timing noise_write_time;
    noise_write_time.start_timing();
    std::ofstream outfile;
    std::string outfile_name = "noise.txt";
    outfile.open(outfile_name);
    for (int i=0; i<num_threads; ++i){
        outfile << noise[i].str();
    }
    outfile.close();
    noise_write_time.end_timing();
    log_text << noise_write_time.print("noise writing");
    logger.write(log_text);

    // Results of the scrub.
    log_text << "Total ticks: " << total_counter
              << "\nNoise: " << Tick::bad_counter
              << "\nPercentage noise: " << 100*double(Tick::bad_counter)/total_counter << "%\n";
    logger.write(log_text);

    //Cleanup
    int rc = munmap(mapped, filesize);
    assert(rc == 0);
    close(fd);

    if (!copy_file_bool) rename(argv[1], "signal.txt");

    program_time.end_timing();
    log_text << program_time.print("total program");
    logger.write(log_text);

    std::cout << "\n";
    return 0;
}
