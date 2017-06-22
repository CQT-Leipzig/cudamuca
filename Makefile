# Makefile for compilation of gcc and cuda version of pmuca ising2D

# please select your mpi compiler
MPICC=mpic++
CPU_FLAGS=-pedantic -Wall -Wextra -O3 -std=c++0x -I./Random123/include/

# please set this to your cuda path
ifeq ($(wildcard /opt/cuda/bin/nvcc),) 
  NVCC=nvcc
else
  NVCC=/opt/cuda/bin/nvcc
endif
GPU_ARCHS=-arch=sm_35 -rdc=true -I./Random123/include/ -lineinfo
GPU_FLAGS=-Xcompiler -Wall,-Wno-unused-function,-O3

all: gpu cpu

gpu: ising2D_gpu

cpu: ising2D_cpu

ising2D_gpu: ising2D.cu
	$(NVCC) $(GPU_ARCHS) $(GPU_FLAGS) --ptxas-options=-v  ising2D.cu -o $@

ising2D_cpu: ising2D.cpp
	$(MPICC) $(CPU_FLAGS) ising2D.cpp -o $@

clean:
	rm -f ising2D_gpu
	rm -f ising2D_cpu
