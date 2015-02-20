#include <iostream>
#include <chrono>
#include <ctime>

std::string now()
{
    //set time_point to current time
    std::chrono::time_point<std::chrono::system_clock> time_point;
    time_point = std::chrono::system_clock::now();

    std::time_t ttp = std::chrono::system_clock::to_time_t(time_point);
    return std::ctime(&ttp);
}

int main(){

    std::cout << "time: " << now() << "\n";

    return 0;
}
