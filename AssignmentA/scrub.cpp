#include <fstream>
#include <string>
#include <cstdlib>
#include <sstream>
#include <thread>
#include <vector>
#include "tick.h"
#include "timing.h"

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

void process_data(char *mapped, int start_location, int end_location)
{
    int location = start_location;
    std::stringstream ss;

    // find newline
    // maybe need to add to bad counter
    while (mapped[location] != '\n' && mapped[location] != '\0' && location != 0){
        ++location;
    }
    while (mapped[location] == '\n') ++location;

    // loop through entire file.
    while (location < end_location){
        // grab the entire line
        while (mapped[location] != '\n' && mapped[location] != '\0'){
            ss << mapped[location];
            ++location;
        }
        std::string test_str = ss.str();
        Tick test(test_str);
        test.check_data();
        ss.str(std::string()); // Empty ss
        ++location;
    }
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
        std::cout << i << ": " << file_split[i] << "\n";
    }
//    for (int i=0; i<num_threads; ++i){
//        process_data(mapped, file_split[i], file_split[i+1]);
//    }
    //process_data(mapped, 0, filesize);

    std::thread thread[num_threads];

    // Launch threads.
    for (int i = 0; i < num_threads; ++i) {
        thread[i] = std::thread(process_data, mapped, file_split[i], file_split[i+1]);
    }
    std::cout << num_threads << " threads launched." << std::endl;

    // Join threads to the main thread of execution.
    for (int i = 0; i < num_threads; ++i) {
        thread[i].join();
    }
    // Even though threads ran independently and asynchronously,
    // output the results as though they had run in serial fashion.
    //for (int i = 0; i<num_threads; i++) std::cout << output[i];

    std::cout << "filesize: " << filesize << "\n";
    std::cout << "counter: " << Tick::counter << "\nbad_counter: " << Tick::bad_counter << "\n";


    //Cleanup
    int rc = munmap(mmappedData, filesize);
    assert(rc == 0);
    close(fd);

    unsigned int n = std::thread::hardware_concurrency();
    std::cout << n << " concurrent threads are supported.\n";

    program_time.end_timing();
    program_time.print("total program");

    return 0;
}

// to do:
// threading
// starting off
// the window thing
