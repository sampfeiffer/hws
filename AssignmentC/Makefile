CPPFLAGS = -std=c++11
GCC = g++
MPICC = mpicxx

all: generate_data.out mpi_mapreduce.out
generate_data.out: generate_data.cpp parameters.h
	$(GCC) $(CPPFLAGS) generate_data.cpp -o generate_data.out

mpi_mapreduce.o: mpi_mapreduce.cpp parameters.h logging.h
	$(MPICC) -c $(CPPFLAGS) mpi_mapreduce.cpp

mpi_mapreduce.out: mpi_mapreduce.o
	$(MPICC) -o mpi_mapreduce.out mpi_mapreduce.o

clean:
	rm -f *.o *.out
