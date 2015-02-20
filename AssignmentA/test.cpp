#include <iostream>
#include <cmath>

void test(int *i)
{
    *i = 5;
}

int main() {

    int foo[3] = {0,0,0};
    std::cout << foo[2] << "\n";

    test(&foo[2]);
    std::cout << foo[2] << "\n";


    return 0;
}
