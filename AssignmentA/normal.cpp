#include <thread>
#include <numeric>
#include "normal_tick.h"
#include "timing.h"
#include "normal_process.h"

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
    char* mapped = static_cast<char*>(mmap(NULL, filesize, PROT_READ, MAP_PRIVATE | MAP_POPULATE, fd, 0));
    assert(mapped != NULL);

    //split up file.
    int num_threads = atoi(argv[2]);
    std::cout << "num_threads: " << num_threads << "\n";
    std::vector<int> file_split(num_threads+1);
    for (int i=0; i<num_threads+1; ++i){
        file_split[i] = i*filesize/num_threads;
    }

    std::thread thread[num_threads];

    std::vector<int> jb_vector(num_threads,0);

    // Launch threads.
    for (int i=0; i<num_threads; ++i) {
        thread[i] = std::thread(normal_process_data, mapped, file_split[i], file_split[i+1], &jb_vector[i]);
    }

    // Join threads to the main thread of execution.
    for (int i=0; i<num_threads; ++i){
        thread[i].join();
    }

    int jarque_bera =std::accumulate(jb_vector.begin(),jb_vector.end(),0);
    float prob = exp(-jarque_bera/2.0);

    std::cout << "total jb: " << jarque_bera << "\n";
    std::cout << "Probability that the returns are normally distributed: " << prob << "\n";

    std::cout << "filesize: " << filesize << "\n";

    //Cleanup
    int rc = munmap(mapped, filesize);
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
// log file
// error catching for command line input
