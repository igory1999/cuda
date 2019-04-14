#include <iostream>
#include <curand.h>

struct random_d_array
{
  float *d_a;
  int n;

  random_d_array(int n) :n{n}
  {
    cudaMalloc((void**)&d_a, n*sizeof(float));
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandGenerateUniform(gen, d_a, n);
  }
  
  ~random_d_array()
  {
    cudaFree(&d_a);
  }
};


using namespace std;

__global__ void MyKernel(float *array, int arrayCount)
{
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if(idx < arrayCount)
    array[idx] *= array[idx];
}

int launchMyKernel(float *array, int arrayCount)
{
  int blockSize;
  int minGridSize;
  int gridSize;
  cudaEvent_t start, stop;
  float milliseconds = 0;  

  blockSize = 32;
  gridSize = (arrayCount + blockSize - 1)/blockSize;
  cout << "Trying non-optiomal blockSize = " << blockSize << ", gridSize = " << gridSize << endl;

  float average = 0.0;
  for(int i = 0; i < 10; ++i)
    {
      cudaEventCreate(&start);
      cudaEventCreate(&stop);
      cudaEventRecord(start);
      MyKernel<<<gridSize, blockSize>>>(array, arrayCount);
      cudaEventRecord(stop); 
      cudaDeviceSynchronize();
      cudaEventElapsedTime(&milliseconds, start, stop);
      cudaEventDestroy(start);
      cudaEventDestroy(stop);
      cout <<"i = "<< i << ": " <<  milliseconds << " ms" << endl;
      if(i > 0) average += milliseconds;
    }
  average /= 10 - 1;
  cout << "Average = " << average << endl;
  cout << "============" << endl;



  cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, 
				     (void*)MyKernel, 0, arrayCount);

  gridSize = (arrayCount + blockSize - 1)/blockSize;

  cout << "Suggested blockSize = " << blockSize << ", gridSize = " << gridSize << ", minGridSize = " << minGridSize << endl;

  average = 0.0;
  for(int i = 0; i < 10; ++i)
    {
      cudaEventCreate(&start);
      cudaEventCreate(&stop);
      cudaEventRecord(start);
      MyKernel<<<gridSize, blockSize>>>(array, arrayCount);
      cudaEventRecord(stop); 
      cudaDeviceSynchronize();
      cudaEventElapsedTime(&milliseconds, start, stop);
      cudaEventDestroy(start);
      cudaEventDestroy(stop);
      cout << "i = " << i << ": " <<  milliseconds << " ms" << endl;
      if(i > 0) average += milliseconds;
    }
  average /= 10 - 1;
  cout << "Average = " << average << endl;
  return 0;
}

int main()
{
  int n = 100000;
  random_d_array A(n);
  launchMyKernel(A.d_a, n);
}
