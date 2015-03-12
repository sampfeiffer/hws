CPPFLAGS = -std=c++11
GCC = g++

all: generate_data
generate_data: generate_data.cpp
	$(GCC) $(CPPFLAGS) generate_data.cpp -o generate_data.out

clean:
	rm -f *.o *.out
