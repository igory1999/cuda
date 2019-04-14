#include <iostream>


__device__ int counter = 0;

__global__ void increment()
{
  atomicAdd(&counter, 1);
}

__global__ void print()
{
  printf("counter = %d\n", counter);
}

int main()
{
  const int blockSize=1024;
  const int gridSize=1024;
  increment<<<blockSize, gridSize>>>();
  print<<<1,1>>>();
  cudaDeviceSynchronize();
}
