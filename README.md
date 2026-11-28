
# GPU-Accelerated Parallel Sorting Using CUDA

<p align="center">
  <b>Performance Analysis of Parallel Sorting Algorithms on GPU</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/CUDA-GPU%20Computing-76B900?logo=nvidia" />
  <img src="https://img.shields.io/badge/C++-Programming-00599C?logo=cplusplus" />
  <img src="https://img.shields.io/badge/Platform-Windows-blue" />
  <img src="https://img.shields.io/badge/Project-Parallel%20Computing-purple" />
</p>

## Overview

This project explores GPU-based parallel sorting using NVIDIA CUDA. It implements and evaluates parallel sorting algorithms and compares their performance against traditional CPU sorting.

The main objective is to understand how GPU parallelism can accelerate sorting operations and analyze the performance differences between custom CUDA implementations, optimized GPU libraries, and CPU-based sorting.

Three approaches are evaluated:

- **Custom Bitonic Sort:** A parallel sorting algorithm implemented using CUDA.
- **NVIDIA Thrust Sort:** An optimized GPU sorting implementation using the Thrust library.
- **CPU std::sort():** A standard C++ sorting implementation used as the baseline.

The project evaluates sorting performance across different input sizes and examines the impact of GPU computation and memory transfer overhead.

## Objectives

- Implement a parallel Bitonic Sorting algorithm using CUDA.
- Explore GPU sorting using NVIDIA Thrust.
- Establish a CPU sorting baseline using `std::sort()`.
- Compare execution time across different input sizes.
- Analyze GPU computation and data transfer overhead.
- Evaluate the scalability of GPU-based sorting.

## Technologies Used

| Technology | Purpose |
|---|---|
| C++ | Core programming language |
| NVIDIA CUDA | GPU parallel computing |
| NVIDIA Thrust | Optimized GPU sorting |
| CUDA Runtime API | GPU memory and execution management |
| Python | Performance graph generation |
| Matplotlib | Data visualization |
| Git & GitHub | Version control |

## Algorithms Implemented

### 1. Custom Bitonic Sort

Bitonic Sort is a comparison-based parallel sorting algorithm particularly suitable for GPU architectures.

It works through multiple stages of comparison and exchange operations.

**Working principle:**

1. Generate a bitonic sequence.
2. Perform parallel compare-and-swap operations.
3. Gradually reduce the comparison distance.
4. Repeat until the complete sequence is sorted.

The implementation uses CUDA threads to execute independent comparison operations in parallel.

**Time complexity:**

- Sequential complexity: O(n log² n)
- Parallel depth: O(log² n)

### 2. NVIDIA Thrust Sort

Thrust is a C++ parallel algorithms library provided by NVIDIA.

The project uses Thrust's GPU sorting functionality to evaluate the performance of an optimized library implementation.

Advantages:

- High-level GPU programming interface.
- Optimized parallel algorithms.
- Efficient memory management.
- Reduced implementation complexity.

### 3. CPU Sorting Using std::sort()

The standard C++ sorting function is used as the CPU baseline.

It provides a reference point for evaluating the performance of GPU-based sorting approaches.

## Experimental Setup

The experiments evaluate sorting performance using three different input sizes.

| Parameter | Configuration |
|---|---|
| Programming Language | C++ |
| GPU Programming | NVIDIA CUDA |
| Custom Algorithm | Bitonic Sort |
| GPU Library | NVIDIA Thrust |
| CPU Baseline | std::sort() |
| Input Sizes | 65,536; 1,048,576; 16,777,216 |
| Visualization | Python Matplotlib |

## Performance Results

The following results were recorded during experimentation.

| Input Size | Thrust Total Time (ms) | CPU Sort Time (ms) |
|---:|---:|---:|
| 65,536 | 0.2361 | 7.9595 |
| 1,048,576 | 2.1698 | 159.418 |
| 16,777,216 | 23.3512 | 2726.66 |

### Performance Analysis

The experimental results demonstrate that NVIDIA Thrust achieves significantly lower recorded execution times than the CPU `std::sort()` implementation across all three tested input sizes.

As the input size increases, the performance difference becomes more pronounced.

This highlights the potential of GPU parallelism for computationally intensive sorting workloads.

However, GPU performance should also be evaluated by considering:

- Host-to-device memory transfer.
- Device-to-host memory transfer.
- Kernel execution time.
- Memory allocation overhead.
- Synchronization overhead.

**Note:** Total execution time and kernel-only execution time represent different measurements. They should not be compared interchangeably.

## Performance Visualization

The project includes generated performance graphs to illustrate the experimental results.

### Graph 1: Sorting Performance Comparison

<!-- Add your actual graph filename below -->
![Sorting Performance Comparison](Graphs/sorting_performance.png)

### Graph 2: Performance Analysis

<!-- Add your actual graph filename below -->
![Performance Analysis](Graphs/performance_analysis.png)

*Ensure that the image filenames match the actual files in the Graphs directory.*

## Correctness Verification

The implementations are evaluated to ensure that the resulting arrays are correctly sorted.

Correctness verification checks whether:

- Elements are arranged in ascending order.
- The sorting operation produces valid output.
- The GPU implementations produce correctly sorted results.

All three recorded input sizes passed correctness verification.

## Project Structure

```text
Parallel-Sorting/
│
├── bitonic_sort.cu
├── thrust_sort.cu
├── cpu_sort.cpp
│
├── Graphs/
│   ├── sorting_performance.png
│   └── performance_analysis.png
│
├── Screenshots/
│   └── execution screenshots
│
├── generate_graphs.py
│
└── README.md
```

*The directory structure above is illustrative. Adjust filenames to match the actual repository.*

## Compilation and Execution

### Prerequisites

- NVIDIA CUDA Toolkit
- CUDA-compatible NVIDIA GPU
- CUDA-capable C++ compiler
- Python 3
- Matplotlib

### Compile CUDA Programs

```powershell
nvcc bitonic_sort.cu -o bitonic_sort.exe
```

```powershell
nvcc thrust_sort.cu -o thrust_sort.exe
```

### Compile CPU Program

```powershell
g++ cpu_sort.cpp -o cpu_sort.exe
```

### Execute Programs

```powershell
.\bitonic_sort.exe
```

```powershell
.\thrust_sort.exe
```

```powershell
.\cpu_sort.exe
```

### Generate Performance Graphs

```powershell
python generate_graphs.py
```

## Key Observations

1. GPU-based sorting can provide substantial performance improvements over CPU sorting for the tested workloads.
2. NVIDIA Thrust provides an optimized GPU sorting implementation.
3. Custom Bitonic Sort demonstrates how parallel comparison operations can be mapped to CUDA threads.
4. Memory transfer overhead plays an important role in overall GPU execution time.
5. Performance depends on input size, algorithm design, GPU architecture, and memory management.

## Limitations

- Performance measurements depend on the hardware and execution environment.
- GPU memory transfer overhead can influence total execution time.
- The custom Bitonic Sort algorithm has different scalability characteristics from library-optimized sorting.
- Further experimentation is required to evaluate performance across additional GPU architectures and input distributions.

## Future Enhancements

- Optimize the custom Bitonic Sort implementation.
- Experiment with additional parallel sorting algorithms.
- Evaluate different input distributions.
- Perform repeated benchmark runs and calculate average execution times.
- Compare kernel execution time against end-to-end execution time.
- Investigate memory coalescing and shared memory optimization.
- Extend benchmarking to larger datasets where hardware capacity permits.

## Conclusion

This project demonstrates the implementation and performance evaluation of parallel sorting using NVIDIA CUDA.

By comparing a custom Bitonic Sort implementation, NVIDIA Thrust, and CPU `std::sort()`, the project explores the advantages and limitations of GPU-based parallel computation.

The experimental results show that NVIDIA Thrust achieves significantly lower recorded execution times than the CPU baseline for the tested input sizes.

The project provides practical experience in CUDA programming, parallel algorithm design, GPU memory management, performance benchmarking, and experimental analysis.

---

<p align="center">
  <b>Developed as a Parallel Computing Project</b>
</p>
