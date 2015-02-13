#include <cstdlib>
#include "tick.h"

int main(int argc, char *argv[])
{
    std::string test_str;
    test_str = "20140804:10:00:00.574914,1173.56,471577";

    Tick test(test_str);
    test.print();

    return 0;
}
