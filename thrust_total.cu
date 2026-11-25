
#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <chrono>

#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/copy.h>

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

int main()
{
    std::vector<int> sizes = {
        1 << 16,
        1 << 20,
        1 << 24
    };

    std::cout << "\nTHRUST TOTAL GPU PIPELINE TIMING\n";

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

        // Allocate GPU memory before timing
        thrust::device_vector<int> gpuData(N);

        // Warm-up
        thrust::sort(gpuData.begin(), gpuData.end());
        CHECK(cudaDeviceSynchronize());

        // Complete pipeline timing
        auto start = std::chrono::high_resolution_clock::now();

        // Host to GPU transfer
        thrust::copy(input.begin(), input.end(),
                     gpuData.begin());

        // GPU sorting
        thrust::sort(gpuData.begin(), gpuData.end());

        // GPU to host transfer
        thrust::copy(gpuData.begin(), gpuData.end(),
                     output.begin());

        CHECK(cudaDeviceSynchronize());

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

        if (!correct)
            return 1;
    }

    return 0;
}
