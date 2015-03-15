#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>
#include <ctype.h>
#include <string.h>
#include <sstream>

int main(int argc, char **argv){
    int line_length = 9;
    int rank, size, ierr;
    char* filename = "tick_data1.dat";
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

    /* figure out who reads what */
    MPI_File_get_size(infile, &filesize);
    filesize--;  /* get rid of text file eof */
    mysize = filesize/size;
    globalstart = rank * mysize;
    globalend = globalstart + mysize - 1;
    if (rank == size-1) globalend = filesize-1;
    if (rank != 0) --globalstart;

    mysize =  globalend - globalstart + 1;

    /* allocate memory */
    chunk = new char[mysize + 1];

    /* everyone reads in their part */
    MPI_File_read_at_all(infile, globalstart, chunk, mysize, MPI_CHAR, MPI_STATUS_IGNORE);
    chunk[mysize] = '\0';

    int locstart=0, locend=mysize-1;
    if (rank != 0) {
        while(chunk[locstart] != '\n') locstart++;
        locstart++;
    }
    while(chunk[locend] != '\n') locend--;
    mysize = locend-locstart+1;


    MPI_File_close(&infile);
    MPI_Finalize();

    std::stringstream strValue;
    float intVal;
    strValue << chunk;
    std::cout << strValue.str();

//    while (!strValue.eof()){
//        strValue >> intVal;
//        std::cout << rank << " " << intVal << "\n";
//    }



    return 0;
}





