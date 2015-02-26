CPPFLAGS = -std=c++11 -pthread
GCC = g++
all: generate_data
generate_data: generate_data.cpp
	$(GCC) $(CPPFLAGS) generate_data.cpp -o generate_data.out

clean:
	rm generate_data.out
	rm -f *.o



#*******************************************************************************
#
# ... (C) 2015, Andrew Sheppard, all rights reserved.
#
#*******************************************************************************