#include <iostream>
#include <thread>

static const int NUM_THREADS = 10;

struct Test
{
    int num;
    thread_local static int num_stat;
    Test(int num_){
        num = num_;
        if (num_stat==0) num_stat = num_;
    }
};
thread_local int Test::num_stat = 0;

void ThreadFunction(int threadID) {
    Test obj(threadID);
    std::cout << obj.num << " " << obj.num_stat << "\n";
}

int main() {
    std::thread thread[NUM_THREADS];

    // Launch threads.
    for (int i = 0; i < NUM_THREADS; ++i) {
        thread[i] = std::thread(ThreadFunction, i);
    }
    std::cout << NUM_THREADS << " threads launched." << std::endl;

    // Join threads to the main thread of execution.
    for (int i = 0; i < NUM_THREADS; ++i) {
        thread[i].join();
    }

    return 0;
}
