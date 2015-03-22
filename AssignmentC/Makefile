CPPFLAGS = -std=c++11
GCC = g++
MPICC = mpicxx

all: generate_data.out mpi_mapreduce.out

generate_data.out: generate_data.cpp parameters.h
	$(GCC) $(CPPFLAGS) generate_data.cpp -o generate_data.out

mpi_mapreduce.o: mpi_mapreduce.cpp parameters.h logging.h timing.h
	$(MPICC) -c $(CPPFLAGS) mpi_mapreduce.cpp

mpi_mapreduce.out: mpi_mapreduce.o
	$(MPICC) -o mpi_mapreduce.out mpi_mapreduce.o

hadoop_mapreduce:
	/usr/bin/g++  -I/usr/lib/hadoop-0.20-mapreduce/include -L/usr/lib/hadoop-0.20-mapreduce/lib/native/Linux-amd64-64 hadoop_mapreduce.cpp -Wall -lhadooppipes -lhadooputils -lpthread -lcrypto -g -O2 -o hadoop_mapreduce.out
	hadoop fs -rm bin/*
	hadoop fs -put hadoop_mapreduce.out bin
	hadoop fs -rm -f -r out-dir

clean:
	rm -f *.o *.out

clean_dat:
	rm -f *.dat
