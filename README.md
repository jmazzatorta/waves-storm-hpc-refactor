# Waves Storm HPC Refactor

This project simulates high-energy particle storms hitting a surface. 

I took a slow, basic code and made it extremely fast. I wrote two new versions for High-Performance Computing (HPC). One version is for a CPU cluster. The other version is for a GPU. Both versions fit the hardware perfectly. They run thousands of times faster than the original code.

---

## Performance Results

I tested the code on AMD EPYC CPUs and an NVIDIA RTX Quadro 6000 GPU. 

**CUDA (1 GPU): 4987x Faster.** The GPU gets more efficient when the problem gets bigger. It uses all 4608 cores at the same time to hide delays.

**MPI + OpenMP (CPU Cluster): 4043x Faster.** This version divides huge data into tiny pieces. These pieces fit inside the fast CPU cache memory. This makes the code run even faster than theoretically expected.

**The "Tie"**
On a small problem, one GPU finishes in 26 milliseconds. A cluster of 128 CPU cores finishes in 25 milliseconds. One GPU is as powerful as four full computers combined.

---

## How It Works

### GPU Acceleration (CUDA)
This code uses thousands of threads at once. It moves active particles into a small, super-fast memory on the chip. Threads read memory in a special grouped order to save time. It also uses special hardware instructions and pre-calculated tables to do math much faster.

### Distributed Cluster (MPI + OpenMP)
This code respects the physical layout of the CPU. MPI splits the big problem between computers. OpenMP splits the smaller problems between the cores. Threads only work on exact chunks of memory. This stops cores from fighting over the same memory line. The code also uses AVX instructions to process eight numbers at the exact same time.

---

## 📂 Project Folders

**Root Folder:** Holds the main C and C++ code. It also holds the `Makefile`.
**`report/`:** Holds my full academic paper. It also holds the Python code used to draw the charts.
**`scripts/`:** Holds the files used to launch jobs on the cluster and check performance.
**`test_files/`:** Holds the test data. It includes a Python script to create custom storm data.

---

## 💻 Technologies Used
**Languages:** C, C++
**Tools:** NVIDIA CUDA, MPI, OpenMP
**Compilers:** GCC, NVCC

---

## ⚙️ How to Build and Run

I made it very easy to build the code. You do not need to type long compiler commands. The `Makefile` and `student.mk` files handle the heavy optimizations automatically.

**To Build:**
Type `make all` to build every version. 
Type `make clean` to delete the built files.

**To Run:**
The `Makefile` has commands ready to run the tests. It uses the exact thread and memory settings from `student.mk`.

Type `make run_mpi` to test the cluster code. It runs on 64 processes with 4 threads each.
Type `make run_cuda` to test the GPU code.
