#include <iostream>

using namespace std;

__device__ double counter = 0.5;


__device__ double myAtomicAdd(double * address, double val)
{
  unsigned long long int * address_as_ull = 
    (unsigned long long int*)address;
  unsigned long long int old = *address_as_ull, assumed;
  do {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed,
		    __double_as_longlong(val + __longlong_as_double(assumed)));
  } while (assumed != old);

  return __longlong_as_double(old);
}


__global__ void increment(double by)
{
  myAtomicAdd(&counter, by);
}

__global__ void print()
{
  printf("counter = %f\n", counter);
}

int main()
{
  const int blockSize=1024;
  const int gridSize=1024;
  const double by = 1.7;

  increment<<<blockSize, gridSize>>>(by);
  print<<<1,1>>>();
  cudaDeviceSynchronize();
  printf("The correct answer is %f\n", (0.5 + by*blockSize*gridSize));
}
