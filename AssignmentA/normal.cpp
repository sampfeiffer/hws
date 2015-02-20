#include <thread>
#include <numeric>
#include <fstream>
#include <sstream>
#include "normal_tick.h"
#include "timing.h"
#include "normal_process.h"
#include "logging.h"

// Memory mapping headers.
#include <sys/mman.h>
#include <sys/types.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <assert.h>

size_t getFilesize(const char* filename) {
    struct stat st;
    stat(filename, &st);
    return st.st_size;
}

int main(int argc, char *argv[])
{
    // Start timing
    Timing program_time;
    program_time.start_timing();

    std::stringstream log_text;
    Logging logger("log_file_normal.txt");

    if (argc < 3){
        std::cout << "ERROR: program needs 2 parameters\n"
                  << "Format is\n"
                  << "     ./normal.out [input_data_filename] [num_of_threads]\n";
        return 1;
    }

    unsigned int filesize = getFilesize(argv[1]);
    int fd = open(argv[1], O_RDONLY, 0); //Open file
    assert(fd != -1); //Error check
    char* mapped = static_cast<char*>(mmap(NULL, filesize, PROT_READ, MAP_PRIVATE | MAP_POPULATE, fd, 0)); //Execute mmap
    assert(mapped != NULL); //Error check

    unsigned int num_threads = atoi(argv[2]);
    unsigned int possible_threads = std::thread::hardware_concurrency();
    log_text << possible_threads << " concurrent threads are supported.\n";
    log_text << "Number of threads: " << num_threads << "\n";
    log_text << "Filesize: " << filesize << "\n";
    logger.write(log_text);

    // Split up file. between threads
    std::vector<int> file_split(num_threads+1);
    for (int i=0; i<num_threads+1; ++i){
        file_split[i] = i*filesize/num_threads;
    }

    std::thread thread[num_threads];
    std::vector<int> jb_vector(num_threads,0);

    // Launch threads.
    for (int i=0; i<num_threads; ++i) {
        thread[i] = std::thread(normal_process_data, mapped, file_split[i], file_split[i+1], &jb_vector[i]);
        log_text << "Thread " << i << " launched\n";
        logger.write(log_text);
    }

    // Join threads to the main thread of execution.
    for (int i=0; i<num_threads; ++i){
        thread[i].join();
    }
    log_text << "Back in main thread.\n";
    logger.write(log_text);

    int jarque_bera =std::accumulate(jb_vector.begin(),jb_vector.end(),0);
    float prob = exp(-jarque_bera/2.0);

    log_text << "Jarque_bera value: " << jarque_bera << "\n";
    logger.write(log_text);
    log_text << "Probability that the returns are normally distributed: " << prob << "\n";
    std::cout << log_text.str() << "\n";
    logger.write(log_text);

    //Cleanup
    int rc = munmap(mapped, filesize);
    assert(rc == 0);
    close(fd);

    program_time.end_timing();
    log_text << program_time.print("total program");
    logger.write(log_text);

    std::cout << "\n";
    return 0;
}
