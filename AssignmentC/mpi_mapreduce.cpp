#include <mpi.h>
#include <stdlib.h>
#include <math.h>

#include "logging.h"
#include "timing.h"
#include "get_drift.h"

int main(int argc, char **argv){
    float get_drift(char* tick_data_filename, int &chars_per_line);
    const char* parameters_filename = "parameters.txt";
    Parameters params(parameters_filename);
    Timing program_timer("whole program");

    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Logging info
    Logging logger("logging.dat");
    if (rank == 0){
        logger.start_info(params, argc, argv, size);
    }

    MPI_File infile;
    int ierr;

    ierr = MPI_File_open(MPI_COMM_WORLD, params.tick_data_filename, MPI_MODE_RDONLY, MPI_INFO_NULL, &infile);
    if (ierr) {
        if (rank == 0) fprintf(stderr, "Couldn't open file %s. Make sure to generate data first.\n", params.tick_data_filename);
        MPI_Finalize();
        exit(1);
    }

    MPI_Offset localsize;
    MPI_Offset globalstart;
    int mysize;
    char *chunk;

    MPI_Offset globalend;
    MPI_Offset filesize;

    // figure out who reads what
    MPI_File_get_size(infile, &filesize);
    mysize = filesize/size;
    globalstart = rank * mysize;
    globalend = globalstart + mysize - 1;
    if (rank != 0) --globalstart;
    mysize =  globalend - globalstart + 1;

    chunk = new char[mysize + 1];

    // All threads read in their part
    MPI_File_read_at_all(infile, globalstart, chunk, mysize, MPI_CHAR, MPI_STATUS_IGNORE);
    chunk[mysize] = '\0';

    // Figure out the start and end of the usable text
    int locstart=0, locend=mysize-1;
    if (rank != 0) {
        while(chunk[locstart] != '\n') locstart++;
        locstart++;
    }
    while(chunk[locend] != '\n') locend--;
    mysize = locend-locstart+1;


    MPI_File_close(&infile);

    std::stringstream str_value;
    float float_value;
    str_value << chunk;
    std::string temp = str_value.str().substr(locstart, mysize);
    str_value.str(std::string());
    str_value << temp;

    int ints_per_thread = round((float(filesize)/params.chars_per_line)/size);

    double sum=0, square_sum=0;

    for (int i=0; i<ints_per_thread; ++i) {
        str_value >> float_value;
        sum += float_value;
        square_sum += float_value*float_value;
    }

    int thread_totals[2] = {int(sum), int(square_sum)};

    if (rank != 0) MPI_Send(&thread_totals, 2, MPI_INT, 0, 0, MPI_COMM_WORLD);
    if (rank == 0){
        double grand_totals[2] = {sum, square_sum};
        int temp[2];
        for (int i=1; i<size; ++i){
            MPI_Recv(&temp, 2, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            grand_totals[0] += temp[0];
            grand_totals[1] += temp[1];
        }

        grand_totals[0] /= 100.0;
        grand_totals[1] /= 10000.0;

        float num_of_data = ints_per_thread*size;
        float mean = grand_totals[0]/num_of_data;
        float stdev = sqrt((grand_totals[1] - (1/num_of_data)*pow(grand_totals[0],2)) / (num_of_data-1));

        std::stringstream ss;
        ss << "num_of_data " << num_of_data << "\n";
        ss << "mean " << mean << "\n";
        ss << "stdev " << stdev << "\n";

        ss << "drift " << get_drift(params.tick_data_filename, params.chars_per_line) << "\n";

        std::cout << ss.str();

        program_timer.end_timing();
        ss << program_timer.print();

        logger.write(ss);
    }

    MPI_Finalize();

    return 0;
}


