#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
int main(void) {

    cudaDeviceProp deviceProp1;
    cudaDeviceProp deviceProp2;

    cudaGetDeviceProperties(&deviceProp1, 0);
    cudaGetDeviceProperties(&deviceProp2, 1);

    printf("\nDevice 0 has %f MB of global RAM, while Device 1 has %f. Cheers!",
            deviceProp1.totalGlobalMem / (1024. * 1024.),
            deviceProp2.totalGlobalMem / (1024. * 1024.));

    return 0;
}
