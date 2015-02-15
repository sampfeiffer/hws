#include <fstream>
#include <string>
#include <cstdlib>
#include <thread>
#include <vector>
#include "tick.h"
#include "timing.h"
#include "process.h"

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
    Timing program_time;
    program_time.start_timing();

    size_t filesize = getFilesize(argv[1]);
    //Open file
    int fd = open(argv[1], O_RDONLY, 0);
    assert(fd != -1);
    //Execute mmap
    void* mmappedData = mmap(NULL, filesize, PROT_READ, MAP_PRIVATE | MAP_POPULATE, fd, 0);
    char* mapped = static_cast<char*>(mmappedData);
    assert(mmappedData != NULL);

    //split up file.
    int num_threads = atoi(argv[2]);
    std::cout << "num_threads: " << num_threads << "\n";
    std::vector<int> file_split(num_threads+1);
    for (int i=0; i<num_threads+1; ++i){
        file_split[i] = i*filesize/num_threads;
    }

    std::thread thread[num_threads];

    // Launch threads.
    for (int i=0; i<num_threads; ++i) {
        thread[i] = std::thread(process_data, mapped, file_split[i], file_split[i+1]);
    }

    // Join threads to the main thread of execution.
    for (int i = 0; i < num_threads; ++i) {
        thread[i].join();
    }

    std::cout << "filesize: " << filesize << "\n";
    std::cout << "counter: " << total_counter << "\nbad_counter: " << Tick::bad_counter << "\n";


    //Cleanup
    int rc = munmap(mmappedData, filesize);
    assert(rc == 0);
    close(fd);

    unsigned int n = std::thread::hardware_concurrency();
    std::cout << n << " concurrent threads are supported.\n";

    program_time.end_timing();
    program_time.print("total program");

    std::cout << "\n";
    return 0;
}

// to do:
// the window thing
// log file
// error catching for command line input
