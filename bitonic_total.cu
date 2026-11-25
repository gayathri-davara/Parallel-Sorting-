
#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <chrono>
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
    std::vector<int> sizes = {
        1 << 16,
        1 << 20,
        1 << 24
    };

    std::cout << "\nBITONIC SORT TOTAL GPU PIPELINE TIMING\n";

    for (int N : sizes)
    {
        std::mt19937 rng(42);
        std::uniform_int_distribution<int> dist(0, 1000000);

        std::vector<int> input(N);

        for (int &x : input)
            x = dist(rng);

        std::vector<int> expected = input;
        std::sort(expected.begin(), expected.end());

        std::vector<int> output(N);

        int *d_data = nullptr;

        CHECK(cudaMalloc((void**)&d_data, N * sizeof(int)));

        int threads = 256;
        int blocks = (N + threads - 1) / threads;

        // Warm-up GPU
        CHECK(cudaMemcpy(d_data, input.data(),
                         N * sizeof(int),
                         cudaMemcpyHostToDevice));

        for (int k = 2; k <= N; k *= 2)
        {
            for (int j = k / 2; j > 0; j /= 2)
            {
                bitonicSort<<<blocks, threads>>>(
                    d_data, j, k, N);
            }
        }

        CHECK(cudaGetLastError());
        CHECK(cudaDeviceSynchronize());

        // Start complete pipeline measurement
        auto start = std::chrono::high_resolution_clock::now();

        // Host to GPU
        CHECK(cudaMemcpy(d_data, input.data(),
                         N * sizeof(int),
                         cudaMemcpyHostToDevice));

        // Bitonic sorting
        for (int k = 2; k <= N; k *= 2)
        {
            for (int j = k / 2; j > 0; j /= 2)
            {
                bitonicSort<<<blocks, threads>>>(
                    d_data, j, k, N);
            }
        }

        CHECK(cudaGetLastError());
        CHECK(cudaDeviceSynchronize());

        // GPU to host
        CHECK(cudaMemcpy(output.data(), d_data,
                         N * sizeof(int),
                         cudaMemcpyDeviceToHost));

        auto stop = std::chrono::high_resolution_clock::now();

        double totalTime =
            std::chrono::duration<double, std::milli>(
                stop - start).count();

        bool correct = (output == expected);

        std::cout << "\nArray Size: " << N << "\n";

        std::cout << "Total GPU Pipeline Time: "
                  << totalTime << " ms\n";

        std::cout << "Verification: "
                  << (correct ? "PASSED" : "FAILED")
                  << "\n";

        cudaFree(d_data);

        if (!correct)
            return 1;
    }

    return 0;
}
