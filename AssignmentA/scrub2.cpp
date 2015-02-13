#include <fstream>
#include <string>
#include <vector>
#include <cstdlib>
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
    //Write the mmapped data to stdout (= FD #1)
    //std::cout << mmappedData << "\n";
    //write(1, mmappedData, 2);
    //std::cout << "\n" << mmappedData << "\n";
    //void *buf;
    //ssize_t read(int fd, void *buf, size_t nbyte);
    int line_start = 0;
    int line_end = line_start;
    while (mapped[line_end] != '\n'){
        std::cout << mapped[line_end];
        ++line_end;
    }
    std::cout << "\n";
    std::cout << line_end << "\n";
    std::cout << mapped[line_end+1] << "\n";
    //Cleanup
    int rc = munmap(mmappedData, filesize);
    assert(rc == 0);
    close(fd);


    std::ifstream infile;
    //std::string filename = argv[1];
    const char * filename = "data10k.txt";
    infile.open(filename);
    std::string test_str;

    //Need to process start data

    int num;

    for (int i=0; i<10000; i++){
        getline(infile, test_str);
        Tick test(test_str);
        //std::cout << test_str << "  ";

        if (num=test.check_data()){
            //std::cout << test_str << "  " << "bad " << num << " " << test.start_time << "\n";
        }
    }
    std::cout << "\ncounter: " << Tick::counter << "\nbad_counter: " << Tick::bad_counter << "\n";


    //test.print();


    infile.close();

    return 0;
}
