CPPFLAGS = -std=c++11
GCC = g++

all: generate_data test
generate_data: generate_data.cpp
	$(GCC) $(CPPFLAGS) generate_data.cpp -o generate_data.out

test: test3.cpp
	mpicxx test3.cpp -o test.out

clean:
	rm -f *.o *.out
