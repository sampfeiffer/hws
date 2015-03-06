CPPFLAGS = -std=c++11
GCC1 = g++
GCC2 = nvcc
all: generate_data cva
generate_data: generate_data.cpp
	$(GCC1) $(CPPFLAGS) generate_data.cpp -o generate_data.out
cva: test_cva2.cu
	$(GCC2) test_cva2.cu -o test.out

clean:
	rm generate_data.out
	rm cva.out
	rm -f *.o



#*******************************************************************************
#
# ... (C) 2015, Andrew Sheppard, all rights reserved.
#
#*******************************************************************************