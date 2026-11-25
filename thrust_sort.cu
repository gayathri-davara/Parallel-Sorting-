
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

    std::cout << "\nTHRUST SORT AND CPU SORT COMPARISON\n";

    for (int N : sizes)
    {
        std::mt19937 rng(42);
        std::uniform_int_distribution<int> dist(0, 1000000);

        std::vector<int> input(N);

        for (int &x : input)
            x = dist(rng);

        // CPU sorting
        std::vector<int> cpuData = input;

        auto cpuStart = std::chrono::high_resolution_clock::now();

        std::sort(cpuData.begin(), cpuData.end());

        auto cpuStop = std::chrono::high_resolution_clock::now();

        double cpuTime =
            std::chrono::duration<double, std::milli>(
                cpuStop - cpuStart).count();

        // Thrust GPU sorting
        thrust::device_vector<int> gpuData(input.begin(), input.end());

        // Warm-up
        thrust::sort(gpuData.begin(), gpuData.end());
        CHECK(cudaDeviceSynchronize());

        // Restore original input before measurement
        thrust::copy(input.begin(), input.end(), gpuData.begin());

        cudaEvent_t start, stop;

        CHECK(cudaEventCreate(&start));
        CHECK(cudaEventCreate(&stop));

        CHECK(cudaEventRecord(start));

        thrust::sort(gpuData.begin(), gpuData.end());

        CHECK(cudaEventRecord(stop));
        CHECK(cudaEventSynchronize(stop));

        float gpuTime = 0;

        CHECK(cudaEventElapsedTime(&gpuTime, start, stop));

        // Copy result back and verify
        std::vector<int> gpuOutput(N);

        thrust::copy(gpuData.begin(), gpuData.end(),
                     gpuOutput.begin());

        bool correct = (gpuOutput == cpuData);

        std::cout << "\nArray Size: " << N << "\n";
        std::cout << "CPU std::sort Time: "
                  << cpuTime << " ms\n";

        std::cout << "Thrust GPU Sort Time: "
                  << gpuTime << " ms\n";

        std::cout << "Verification: "
                  << (correct ? "PASSED" : "FAILED")
                  << "\n";

        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        if (!correct)
            return 1;
    }

    return 0;
}
