\# Theme 8 – Parallel Sorting on the GPU



\## Project Overview



This project implements and evaluates GPU-based parallel sorting using CUDA. It compares a custom Bitonic Sort implementation, NVIDIA Thrust Sort, and CPU-based `std::sort`.



\## Hardware and Software



\- \*\*GPU:\*\* NVIDIA GeForce RTX 5060 Ti

\- \*\*CUDA Toolkit:\*\* 13.4

\- \*\*Compiler:\*\* NVIDIA NVCC with Visual Studio Build Tools

\- \*\*Language:\*\* C++ and CUDA C++



\## Implementations



1\. \*\*Bitonic Sort:\*\* Custom CUDA kernel using parallel compare-exchange operations.

2\. \*\*Thrust Sort:\*\* GPU sorting using NVIDIA's Thrust library.

3\. \*\*CPU Sort:\*\* Standard C++ `std::sort` algorithm.



\## Experimental Results



All timings are in milliseconds.



| Array Size | Bitonic Kernel | Bitonic Total | Thrust Sort | Thrust Total | CPU Sort |

|---:|---:|---:|---:|---:|---:|

| 65,536 | 0.986944 | 1.0062 | 0.095008 | 0.2361 | 7.9595 |

| 1,048,576 | 2.06643 | 3.4772 | 0.582784 | 2.1698 | 159.418 |

| 16,777,216 | 85.2817 | 103.535 | 3.57914 | 23.3512 | 2726.66 |



All sorting implementations passed correctness verification.



\## Performance Analysis



\- Thrust Sort achieved the lowest GPU sorting time in all three tests.

\- The custom Bitonic Sort required more execution time due to its multiple kernel launches.

\- Including data transfers increased GPU execution time.

\- Thrust remained faster than CPU `std::sort` for all tested array sizes, even when including data transfers.



\## Conclusion



The experiment demonstrates how GPU parallelism can accelerate large-scale sorting. It also shows that optimized GPU libraries such as Thrust can outperform a basic custom sorting kernel.



\## Project Files



\- `bitonic\_sort.cu` – Custom CUDA Bitonic Sort.

\- `thrust\_sort.cu` – Thrust Sort and CPU comparison.

\- `bitonic\_total.cu` – Bitonic Sort including data transfers.

\- `thrust\_total.cu` – Thrust Sort including data transfers.

\- `results.csv` – Experimental timing results.

\- `cuda\_test.cu` – Initial CUDA functionality test.

