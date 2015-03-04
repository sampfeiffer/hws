// calculate how many deals i can read in to one gpu.
// create a device_vector of the appropriate amount of counterparties
// run the simulations and cva calculator on the vector of counterparties


struct add
{
    T operator()(T a, T b)
    {
        return a + b;
    }
};

#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

int main(void) {

    cudaDeviceProp deviceProp1;
    cudaDeviceProp deviceProp2;

    cudaGetDeviceProperties(&deviceProp1, 0);
    cudaGetDeviceProperties(&deviceProp2, 1);

    printf("\nDevice 0 has %f MB of global RAM, while Device 1 has %f. Cheers!\n",
            deviceProp1.totalGlobalMem / (1024. * 1024.),
            deviceProp2.totalGlobalMem / (1024. * 1024.));

    int num_gpus=0;
    cudaGetDeviceCount(&num_gpus);
    printf("number of CUDA devices:\t%d\n", num_gpus);

    int x = 10; int y = 20; int z;

    add<int> func; // create an add functor for T=int

    z = func(x,y); // invoke functor on x and y
    float x = 10; float y = 20; float z;
    add<float> func; // create an add functor for T=float
    z = func(x,y); // invoke functor on x and y



    return 0;
}
