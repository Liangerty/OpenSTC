#include "Parallel.h"
#include <cstdio>

void cfd::MpiParallel::setup_gpu_device() const {
  int deviceCount{0};
  cudaGetDeviceCount(&deviceCount);

  if (deviceCount < n_proc) {
    printf("Not enough GPU devices.\n"
           "We want %d GPUs but only %d GPUs are available.\n"
           " Stop computing.\n", n_proc, deviceCount);
    exit();
  }

  cudaDeviceProp prop{};
  cudaGetDeviceProperties(&prop,my_id);
  cudaSetDevice(my_id);
  printf("Process %d will compute on device %s.\n", my_id, prop.name);
}
