CPPFLAGS = -O3 -std=c++11 -pthread
GCC = g++
all: scrub normal
scrub: scrub2.cpp
	$(GCC) $(CPPFLAGS) scrub2.cpp -o scrub2.out
normal: normal.cpp
	$(GCC) $(CPPFLAGS) normal.cpp -o normal

clean:
	rm scrub
	rm -f *.o



#*******************************************************************************
#
# ... (C) 2015, Andrew Sheppard, all rights reserved.
#
#*******************************************************************************