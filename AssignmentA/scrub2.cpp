#include <fstream>
#include <string>
#include <cstdlib>
#include <sstream>
#include <thread>
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

    int location = 0;
    std::stringstream ss;

    // loop through entire file.
    while (mapped[location] != '\0'){
        // grab the entire line
        while (mapped[location] != '\n'){
            if (mapped[location] == '\0') break;
            ss << mapped[location];
            ++location;
        }
        std::string test_str = ss.str();
        Tick test(test_str);
        test.check_data();
        ss.str(std::string()); // Empty ss
        ++location;
    }
    std::cout << "counter: " << Tick::counter << "\nbad_counter: " << Tick::bad_counter << "\n";


    //Cleanup
    int rc = munmap(mmappedData, filesize);
    assert(rc == 0);
    close(fd);

    unsigned int n = std::thread::hardware_concurrency();
    std::cout << n << " concurrent threads are supported.\n";

    program_time.end_timing();
    program_time.print();

    return 0;
}
