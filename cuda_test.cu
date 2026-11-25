
#include <iostream>
#include <cuda_runtime.h>

__global__ void testKernel(int *data)
{
    int tid = threadIdx.x;
    data[tid] = tid * 10;
}

int main()
{
    int h_data[5];
    int *d_data;

    cudaError_t err = cudaMalloc(&d_data, 5 * sizeof(int));

    if (err != cudaSuccess)
    {
        std::cout << cudaGetErrorString(err) << "\n";
        return 1;
    }

    testKernel<<<1, 5>>>(d_data);

    err = cudaDeviceSynchronize();

    if (err != cudaSuccess)
    {
        std::cout << "Kernel error: "
                  << cudaGetErrorString(err) << "\n";
        cudaFree(d_data);
        return 1;
    }

    cudaMemcpy(h_data, d_data, 5 * sizeof(int),
               cudaMemcpyDeviceToHost);

    std::cout << "GPU computation results:\n";

    for (int i = 0; i < 5; i++)
        std::cout << h_data[i] << " ";

    std::cout << "\nCUDA test completed successfully!\n";

    cudaFree(d_data);

    return 0;
}
