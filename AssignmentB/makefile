CPPFLAGS = -std=c++11
GCC1 = g++
GCC2 = nvcc
all: generate_data cva
generate_data: generate_data.cpp
	$(GCC1) $(CPPFLAGS) generate_data.cpp -o generate_data.out
cva: cva.cu
	$(GCC2) cva.cu -o cva.out

clean:
	rm generate_data.out
	rm cva.out
	rm -f *.o



#*******************************************************************************
#
# ... (C) 2015, Andrew Sheppard, all rights reserved.
#
#*******************************************************************************