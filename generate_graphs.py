
import pandas as pd
import matplotlib.pyplot as plt
import os

os.makedirs("Graphs", exist_ok=True)

df = pd.read_csv("results.csv")
x = df["Array_Size"].astype(str)

# Graph 1: GPU Kernel Performance
plt.figure(figsize=(10, 6))

plt.plot(x, df["Bitonic_Kernel_ms"], marker="o",
         label="Bitonic Kernel")
plt.plot(x, df["Thrust_Sort_ms"], marker="o",
         label="Thrust Sort")

plt.yscale("log")
plt.xlabel("Array Size")
plt.ylabel("Execution Time (ms)")
plt.title("GPU Sorting Kernel Performance")
plt.legend()
plt.grid(True, which="both", alpha=0.3)
plt.tight_layout()
plt.savefig("Graphs/gpu_kernel_performance.png", dpi=200)
plt.close()

# Graph 2: Total Pipeline Performance
plt.figure(figsize=(10, 6))

plt.plot(x, df["Bitonic_Total_ms"], marker="o",
         label="Bitonic Total")
plt.plot(x, df["Thrust_Total_ms"], marker="o",
         label="Thrust Total")
plt.plot(x, df["CPU_Sort_ms"], marker="o",
         label="CPU std::sort")

plt.yscale("log")
plt.xlabel("Array Size")
plt.ylabel("Execution Time (ms)")
plt.title("Total Pipeline Performance")
plt.legend()
plt.grid(True, which="both", alpha=0.3)
plt.tight_layout()
plt.savefig("Graphs/total_pipeline_performance.png", dpi=200)
plt.close()

print("Both graphs generated successfully!")
