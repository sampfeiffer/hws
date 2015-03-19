#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>
#include <ctype.h>
#include <string.h>
#include <sstream>
#include <math.h>
#include "parameters.h"

// TODO remember to divide sum by 100 and sum^2 by 10000

int main(int argc, char **argv){
    int line_length = 9;
    int rank, size, ierr;
    char* filename = "tick_data.dat";
    char* parameters_filename = "parameters.txt";
    Parameters params(parameters_filename);
    MPI_File infile;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    ierr = MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_RDONLY, MPI_INFO_NULL, &infile);
    if (ierr) {
        if (rank == 0) fprintf(stderr, "Couldn't open file %s\n", filename);
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

    // allocate memory
    chunk = new char[mysize + 1];

    // All threads read in their part
    MPI_File_read_at_all(infile, globalstart, chunk, mysize, MPI_CHAR, MPI_STATUS_IGNORE);
    chunk[mysize] = '\0';

    int locstart=0, locend=mysize-1;
    if (rank != 0) {
        while(chunk[locstart] != '\n') locstart++;
        locstart++;
    }
    while(chunk[locend] != '\n') locend--;
    mysize = locend-locstart+1;

    int ints_per_thread = round((float(filesize)/params.chars_per_line)/size);

    MPI_File_close(&infile);

    std::stringstream str_value;
    float float_value;
    str_value << chunk;
    std::string temp = str_value.str().substr(locstart, mysize);
    str_value.str(std::string());
    str_value << temp;

    double sum=0, square_sum=0;

    for (int i=0; i<ints_per_thread; ++i) {
        str_value >> float_value;
        sum += float_value;
        square_sum += float_value*float_value;
    }

    int thread_totals[2] = {sum, square_sum};

    std::cout << rank << " " << sum << " " << square_sum << "\n";
    if (rank != 0) MPI_Send(&thread_totals, 2, MPI_INT, 0, 0, MPI_COMM_WORLD);
    if (rank == 0){
        int grand_totals[2] = {sum, square_sum}, temp[2]={0,0};
        for (int i=1; i<size; ++i){
            MPI_Recv(&temp, 2, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            grand_totals[0] += temp[0];
            grand_totals[1] += temp[1];
            std::cout << i << "temp " << temp[0] << " " << temp[1] << "\n";
        }
        grand_totals[0] /= 100;
        grand_totals[1] /= 10000;
        std::cout << "total sum " << grand_totals[0] << " " << grand_totals[1] << "\n";

    }

    MPI_Finalize();

    return 0;
}


