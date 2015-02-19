#include <iostream>
#include <cmath>
#include <sstream>

int main() {
    double time = 48327.324;

    std::stringstream time_text;
    double integral;
    double seconds_frac = modf(time, &integral);

    int time_int = int(time);

    int seconds_int = time_int % 60;
    int minutes = (time_int-seconds_int)/60 % 60;
    int hours = (time_int-minutes*60-seconds_int) / 3600;


    time_text << hours << ":" << minutes << ":" << seconds_int + seconds_frac << "\n";
    std::string test = time_text.str();
    std::cout << test << "\n";

    return 0;
}
