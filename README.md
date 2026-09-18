# Game of Life - CPU, SIMD, CUDA and OpenGL

This repository contains several implementations of Conway's Game of Life, designed to compare performance and parallelization approaches across CPU, SIMD and GPU architectures.

The project is organized into three main areas:

- sequential and SIMD implementations in C
- parallel versions using CUDA
- graphical visualization with OpenGL + CUDA interop

The goal is to evaluate how execution time, resource usage and scalability change in different scenarios using the same computational problem.

---

## Table of contents

- [Project description](#project-description)
- [Repository structure](#repository-structure)
- [Available implementations](#available-implementations)
- [Requirements](#requirements)
- [Build](#build)
- [Execution](#execution)
- [Game of Life rules](#game-of-life-rules)
- [Notes](#notes)

---

## Project description

The Game of Life is a cellular automaton created by John Conway. A two-dimensional grid evolves at each generation according to three simple rules:

- a live cell with fewer than 2 neighbors dies from isolation
- a live cell with more than 3 neighbors dies from overpopulation
- a dead cell with exactly 3 neighbors is born

In this repository, these rules are implemented in different ways:

- standard sequential CPU version
- optimized SIMD/SSE version
- CUDA naive versions and 2D block-based versions
- interactive graphical visualization through OpenGL

---

## Repository structure

```text
game-of-life-cuda-simd/
├── README.md
├── CPU_sequential_&_SIMD/
│   ├── GameOfLife_CPU.c
│   ├── GameOfLife_CPU_SIMD_clock.c
│   ├── GameOfLife_CPU_SIMD_stats.c
│   ├── GameOfLife_CPU_stats.c
│   └── ...
├── CUDA_parallel/
│   ├── gol_naive.cu
│   ├── gol_2D2D_block16x16.cu
│   ├── gol_2D2D_block32x32.cu
│   └── ...
├── Graphics_openGL/
│   ├── gol_naive_opengl.cu
│   ├── gol_2D2D_block16x16_opengl.cu
│   ├── gol_CPUlogic_grafica.cu
│   ├── include/
│   ├── lib/
│   ├── src/
│   └── build/
└── ...
```

---

## Available implementations

### 1) CPU and SIMD

Folder: [CPU_sequential_&_SIMD](CPU_sequential_&_SIMD)

Main files:

- [CPU_sequential_&_SIMD/GameOfLife_CPU.c](CPU_sequential_&_SIMD/GameOfLife_CPU.c): sequential grid implementation using the standard logic
- [CPU_sequential_&_SIMD/GameOfLife_CPU_stats.c](CPU_sequential_&_SIMD/GameOfLife_CPU_stats.c): version focused on collecting statistics and analyzing performance
- [CPU_sequential_&_SIMD/GameOfLife_CPU_SIMD_clock.c](CPU_sequential_&_SIMD/GameOfLife_CPU_SIMD_clock.c): SIMD-optimized version with timing measurement
- [CPU_sequential_&_SIMD/GameOfLife_CPU_SIMD_stats.c](CPU_sequential_&_SIMD/GameOfLife_CPU_SIMD_stats.c): SIMD version focused on performance statistics and comparison

These versions use a 2D grid structure with padding and, in some cases, SSE intrinsics to process multiple cells in parallel inside the SIMD register.

### 2) CUDA

Folder: [CUDA_parallel](CUDA_parallel)

Main files:

- [CUDA_parallel/gol_naive.cu](CUDA_parallel/gol_naive.cu): minimal CUDA kernel, very simple but not highly optimized
- [CUDA_parallel/gol_2D2D_block16x16.cu](CUDA_parallel/gol_2D2D_block16x16.cu): kernel with 2D block organization and 16x16 block size
- [CUDA_parallel/gol_2D2D_block32x32.cu](CUDA_parallel/gol_2D2D_block32x32.cu): variant with 32x32 blocks
- [CUDA_parallel/gol_16x16_shared_mem.cu](CUDA_parallel/gol_16x16_shared_mem.cu): version using shared memory to reduce global memory accesses
- [CUDA_parallel/gol_16x16_sm_lineare.cu](CUDA_parallel/gol_16x16_sm_lineare.cu): variant with linear access and block management

These implementations show the transition from the simplest model to more performant solutions, with particular attention to block sizing and data locality.

### 3) OpenGL + CUDA

Folder: [Graphics_openGL](Graphics_openGL)

Main files:

- [Graphics_openGL/gol_naive_opengl.cu](Graphics_openGL/gol_naive_opengl.cu): visual version using OpenGL and CUDA interop
- [Graphics_openGL/gol_2D2D_block16x16_opengl.cu](Graphics_openGL/gol_2D2D_block16x16_opengl.cu): 2D block-based visualization with graphical rendering
- [Graphics_openGL/gol_CPUlogic_grafica.cu](Graphics_openGL/gol_CPUlogic_grafica.cu): version combining CPU logic and graphical rendering

These versions open a graphical window in which the grid is displayed as an image, making it easier to observe the evolution of generations visually.

---

## Requirements

### For CPU/SIMD versions

- C/C++ compiler
- GCC, Clang or MSVC
- SSE/AVX support depending on the file used
- Windows, Linux or macOS

### For CUDA versions

- NVIDIA GPU compatible with CUDA
- CUDA Toolkit installed
- `nvcc` available in PATH
- updated NVIDIA driver

### For OpenGL versions

- OpenGL libraries
- GLFW
- GLAD
- CUDA Toolkit with interop support

On Windows, Visual Studio toolchains are often used, or a custom `nvcc` + linker configuration with GLAD/GLFW.

---

## Build

### Sequential CPU build

Example with GCC:

```bash
gcc CPU_sequential_&_SIMD/GameOfLife_CPU.c -O3 -msse2 -o GameOfLife_CPU
./GameOfLife_CPU
```

### SIMD build

```bash
gcc CPU_sequential_&_SIMD/GameOfLife_CPU_SIMD_clock.c -O3 -msse2 -o GameOfLife_CPU_SIMD
./GameOfLife_CPU_SIMD
```

### CUDA build

Example:

```bash
nvcc CUDA_parallel/gol_naive.cu -o gol_naive
./gol_naive
```

Or:

```bash
nvcc CUDA_parallel/gol_2D2D_block16x16.cu -o gol_2D2D_block16x16
./gol_2D2D_block16x16
```

### OpenGL build

OpenGL versions usually require a setup with:

- GLAD and GLFW include directories
- OpenGL and GLFW linker libraries
- CUDA interop support for registering the graphics buffer

The exact method depends on the operating system and IDE used. On Windows, this is commonly done inside a Visual Studio project or through a custom `nvcc` + linker setup.

---

## Execution

After compilation, the programs run the simulation for a fixed number of generations. In many versions, the grid is initialized with:

- a known pattern such as a glider
- or a random grid with a deterministic seed

Some files use explicitly set dimensions, such as 32x32, 64x64, 1024x1024 or even 2048x2048, depending on the performance test being run.

---

## Game of Life rules

The logic is as follows:

- a live cell with 2 or 3 neighbors survives
- a live cell with 0, 1, or 4+ neighbors dies
- a dead cell with exactly 3 neighbors is born

This logic is implemented in both sequential and SIMD versions as well as in CUDA kernels.

---

## Notes

- The repository is intended for educational and performance-comparison purposes.
- Some files are experimental versions with optimizations, timing checks and tests of different configurations.
- Performance depends heavily on:
  - grid size
  - number of generations
  - CPU/GPU architecture
  - CUDA block organization
  - use of shared memory or contiguous memory access

---

## Educational objective

The project makes it possible to observe how the same algorithm can be tackled with different approaches:

- CPU serial: simple and easy to read
- CPU + SIMD: greater data-level parallelism
- CUDA: massive parallelism on the GPU
- OpenGL: real-time rendering of the result

---

## Author / usage

Project developed for academic and experimental purposes, focused on analyzing the performance of Game of Life-based algorithms.
