CPPFLAGS = -O3 -std=c++11 -pthread
GCC = g++
all: scrub normal
scrub: scrub.cpp
	$(GCC) $(CPPFLAGS) scrub.cpp -o scrub.out
normal: normal.cpp
	$(GCC) $(CPPFLAGS) normal.cpp -o normal

clean:
	rm scrub.out
	rm -f *.o



#*******************************************************************************
#
# ... (C) 2015, Andrew Sheppard, all rights reserved.
#
#*******************************************************************************