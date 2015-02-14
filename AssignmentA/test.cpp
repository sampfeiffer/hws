#include <iostream>
#include <thread>
#include <string>

static const int NUM_THREADS = 10;
static std::string output[NUM_THREADS];

// Thread function. When a thread is launched, this is the code that
// gets executed.
void ThreadFunction(int threadID) {
    output[threadID] = "Hello from thread #" + std::to_string(threadID) + '\n';
}

int main()
{
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
    // Even though threads ran independently and asynchronously,
    // output the results as though they had run in serial fashion.
    for (int i = 0; i<NUM_THREADS; i++) std::cout << output[i];

    return 0;
}
