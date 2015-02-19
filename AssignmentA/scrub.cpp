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

void copy_file(const char *from, const char *to)
{
    std::ifstream inFile(from);
    std::ofstream outFile(to);

    outFile << inFile.rdbuf();
    inFile.close();
    outFile.close();
}

int main(int argc, char *argv[])
{
    const char *signal_filename = "signal.txt";
    bool copy_file_bool = false;

    // Copy data file into a file called "signal.txt". All operations take place on this file.
    Timing copy_file_time;
    copy_file_time.start_timing();
    if (!argv[3] || (argv[3] && strcmp(argv[3], "no") != 0)) copy_file_bool = true;
    if (copy_file_bool){
        copy_file(argv[1], signal_filename);
    }
    copy_file_time.end_timing();


    Timing program_time;
    program_time.start_timing();

    size_t filesize = getFilesize(signal_filename);
    //Open file
    int fd = open(signal_filename, O_RDWR, 0);
    assert(fd != -1);
    //Execute mmap
    char* mapped = static_cast<char*>(mmap(NULL, filesize, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE, fd, 0));
    assert(mapped != NULL);

    //split up file.
    int num_threads = atoi(argv[2]);
    std::cout << "num_threads: " << num_threads << "\n";
    std::vector<int> file_split(num_threads+1);
    for (int i=0; i<num_threads+1; ++i){
        file_split[i] = i*filesize/num_threads;
    }

    std::thread thread[num_threads];
    std::vector<std::stringstream> noise(num_threads);

    // Launch threads.
    for (int i=0; i<num_threads; ++i) {
        thread[i] = std::thread(process_data, mapped, file_split[i], file_split[i+1], &noise[i]);
    }

    // Join threads to the main thread of execution.
    for (int i=0; i<num_threads; ++i){
        thread[i].join();
    }

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

    std::cout << "filesize: " << filesize << "\n";
    std::cout << "counter: " << total_counter << "\nbad_counter: " << Tick::bad_counter << "\n";


    //Cleanup
    int rc = munmap(mapped, filesize);
    assert(rc == 0);
    close(fd);

    unsigned int n = std::thread::hardware_concurrency();
    std::cout << n << " concurrent threads are supported.\n";

    if (!copy_file_bool) rename(argv[1], "signal.txt");

    program_time.end_timing();

    copy_file_time.print("file copy");
    noise_write_time.print("noise writing");
    program_time.print("total program");

    std::cout << "\n";
    return 0;
}

// to do:
// the window thing
// log file
// error catching for command line input
