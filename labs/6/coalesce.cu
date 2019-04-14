#include <iostream>
#include <curand.h>

using namespace std;

#include <curand.h>

struct random_d_array
{
  float *data;
  int n;

  random_d_array(int n) :n{n}
  {
    cudaMalloc((void**)&data, n*sizeof(float));
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandGenerateUniform(gen, data, n);
  }
  
  ~random_d_array()
  {
    cudaFree(&data);
  }
};

__global__ void copy1(float *a, float *b, int n)
{
  int id = threadIdx.x + blockDim.x * blockIdx.x;
  if(id < n)
    a[id] = b[id];
}

__global__ void copy2(float *a, float *b, int n, int offset)
{
  int id = threadIdx.x + blockDim.x * blockIdx.x;
  if(id < n)
    a[id] = b[(id + offset) % n];
}

__global__ void copy3(float *a, float *b, int n, int stride)
{
  int id = threadIdx.x + blockDim.x * blockIdx.x;
  if(id < n)
    a[id] = b[(id * stride) % n];
}

float call1(float *a, float *b, int n, int blockSize, int gridSize, int i)
{
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);
  copy1<<<gridSize,blockSize>>>(a, b, n);
  cudaEventRecord(stop); 
  cudaEventSynchronize(stop);
  float milliseconds = 0;  
  cudaEventElapsedTime(&milliseconds, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cout << "copy1: " << i << ": " << milliseconds << " ms" << endl;
  return milliseconds;
}

float call2(float *a, float *b, int n, int offset, int blockSize, int gridSize, int i)
{
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);
  copy2<<<gridSize,blockSize>>>(a, b, n, offset);
  cudaEventRecord(stop); 
  cudaEventSynchronize(stop);
  float milliseconds = 0;  
  cudaEventElapsedTime(&milliseconds, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cout << "copy2: " << i << ": " << milliseconds << " ms" << endl;
  return milliseconds;
}

float call3(float *a, float *b, int n, int stride, int blockSize, int gridSize, int i)
{
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);
  copy2<<<gridSize,blockSize>>>(a, b, n, stride);
  cudaEventRecord(stop); 
  cudaEventSynchronize(stop);
  float milliseconds = 0;  
  cudaEventElapsedTime(&milliseconds, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cout << "copy3: " << i << ": " << milliseconds << " ms" << endl;
  return milliseconds;
}


int main()
{
  const int N = 1024*1024*1024;
  const int iterations = 10;
  const int blockSize = 32;
  const int gridSize = (N + blockSize - 1)/blockSize;
  
  random_d_array a(N);
  random_d_array b(N);
  float average = 0.0;

  cout << "============= coalesced read ==============" << endl;
  call1(a.data, b.data, N, blockSize, gridSize, -1);
  for(int i = 0; i < iterations; ++i)
    average += call1(a.data, b.data, N, blockSize, gridSize, i);
  average /= iterations;
  cout << "Average = " << average << endl;
  cout << "============= offset read =================" << endl;

  average = 0.0;
  const int offset = 17;
  call2(a.data, b.data, N, offset, blockSize, gridSize, -1);
  for(int i = 0; i < iterations; ++i)
    average += call2(a.data, b.data, N, offset, blockSize, gridSize, i);
  average /= iterations;
  cout << "Average = " << average << endl;
  cout << "============= strided read ================" << endl;

  average = 0.0;
  const int stride = 17;
  call3(a.data, b.data, N, stride, blockSize, gridSize, -1);
  for(int i = 0; i < iterations; ++i)
    average += call3(a.data, b.data, N, stride, blockSize, gridSize, i);
  average /= iterations;
  cout << "Average = " << average << endl;
}
