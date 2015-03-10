CPPFLAGS1 = -std=c++11
CPPFLAGS2 = -Xcompiler -fopenmp -O3 -arch=sm_20
GCC1 = g++
GCC2 = nvcc
all: generate_data cva
generate_data: generate_data.cpp
	$(GCC1) $(CPPFLAGS1) generate_data.cpp -o generate_data.out
cva: cva.cu
	$(GCC2) $(CPPFLAGS2) -o cva.out cva.cu -lgomp

clean:
	rm generate_data.out
	rm cva.out
	rm -f *.o
