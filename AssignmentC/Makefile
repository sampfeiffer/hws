CPPFLAGS = -std=c++11
GCC = g++

all: generate_data mpi_mapreduce
generate_data: generate_data.cpp
	$(GCC) $(CPPFLAGS) generate_data.cpp -o generate_data.out

mpi_mapreduce: mpi_mapreduce.cpp
	mpicxx mpi_mapreduce.cpp -o mpi_mapreduce.out

clean:
	rm -f *.o *.out
