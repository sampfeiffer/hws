CPPFLAGS = -std=c++11 -pthread
GCC = g++
all: generate_data cva
generate_data: generate_data.cpp
	$(GCC) $(CPPFLAGS) generate_data.cpp -o generate_data.out
cva: cva.cpp
	$(GCC) $(CPPFLAGS) cva.cpp -o cva.out

clean:
	rm generate_data.out
	rm cva.out
	rm -f *.o



#*******************************************************************************
#
# ... (C) 2015, Andrew Sheppard, all rights reserved.
#
#*******************************************************************************