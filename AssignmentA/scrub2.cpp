#include <fstream>
#include <string>
#include <vector>
#include <cstdlib>
#include <sstream>
#include "tick.h"

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

    while (mapped[location] != '\0'){
        while (mapped[location] != '\n'){
            if (mapped[location] == '\0') break;
            ss << mapped[location];
            ++location;
        }
        std::string test_str = ss.str();
        Tick test(test_str);
        test.check_data();
        ss.str(std::string());
        ++location;
    }
    std::cout << "\ncounter: " << Tick::counter << "\nbad_counter: " << Tick::bad_counter << "\n";


    //Cleanup
    int rc = munmap(mmappedData, filesize);
    assert(rc == 0);
    close(fd);

    return 0;
}
