
#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <cuda_runtime.h>

#define CHECK(call) \
{ \
    cudaError_t err = call; \
    if (err != cudaSuccess) \
    { \
        std::cerr << "CUDA Error: " \
                  << cudaGetErrorString(err) << std::endl; \
        return 1; \
    } \
}

__global__ void bitonicSort(int *data, int j, int k, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= n)
        return;

    int partner = i ^ j;

    if (partner > i && partner < n)
    {
        bool ascending = ((i & k) == 0);

        int a = data[i];
        int b = data[partner];

        if ((ascending && a > b) ||
            (!ascending && a < b))
        {
            data[i] = b;
            data[partner] = a;
        }
    }
}

int main()
{
    const int N = 1 << 24;

    std::vector<int> input(N);

    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist(0, 1000000);

    for (int &x : input)
        x = dist(rng);

    std::vector<int> expected = input;
    std::sort(expected.begin(), expected.end());

    int *d_data;

    CHECK(cudaMalloc((void**)&d_data, N * sizeof(int)));

    CHECK(cudaMemcpy(d_data, input.data(),
                     N * sizeof(int),
                     cudaMemcpyHostToDevice));

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    cudaEvent_t start, stop;

    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));

    CHECK(cudaEventRecord(start));

    for (int k = 2; k <= N; k *= 2)
    {
        for (int j = k / 2; j > 0; j /= 2)
        {
            bitonicSort<<<blocks, threads>>>(d_data, j, k, N);
        }
    }

    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));

    float time = 0;
    CHECK(cudaEventElapsedTime(&time, start, stop));

    std::vector<int> output(N);

    CHECK(cudaMemcpy(output.data(), d_data,
                     N * sizeof(int),
                     cudaMemcpyDeviceToHost));

    bool correct = (output == expected);

    std::cout << "\nBITONIC SORT - CUDA\n";
    std::cout << "Array Size: " << N << "\n";
    std::cout << "GPU Kernel Time: " << time << " ms\n";
    std::cout << "Verification: "
              << (correct ? "PASSED" : "FAILED") << "\n";

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_data);

    return correct ? 0 : 1;
}
