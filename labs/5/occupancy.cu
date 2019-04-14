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

__global__ void MyKernel(float *d, float *a, float *b, int n)
{
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if(idx < n)
    d[idx] = a[idx] * b[idx];
}

int main()
{
  int numBlocks;
  int blockSize = 32;
  
  int device;
  cudaDeviceProp prop;
  int activeWarps;
  int maxWarps;

  int N = 1024*1024;
  random_d_array a(N);
  random_d_array b(N);
  random_d_array d(N);

  string buffer;
  while(true)
    {
      cout << "Enter the block size or q to exit" << endl;
      cin >> buffer;
      if(buffer == "q") break;
      blockSize = stoi(buffer, nullptr);

      int gridSize = (N + blockSize - 1)/blockSize;

      cout << "blockSize = " << blockSize << ", gridSize = " << gridSize << endl;

      cudaGetDevice(&device);
      cudaGetDeviceProperties(&prop, device);
      
      cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocks, MyKernel, blockSize, 0);
      
      activeWarps = numBlocks * blockSize/prop.warpSize;
      maxWarps = prop.maxThreadsPerMultiProcessor/prop.warpSize;

      cout << "Occupancy: " << (double)activeWarps/maxWarps * 100 << "%" << endl;

      double average = 0.0;
      int iterations = 5;
      
      for(int i = 0; i < iterations; ++i)
	{
	  cudaEvent_t start, stop;
	  cudaEventCreate(&start);
	  cudaEventCreate(&stop);
	  cudaEventRecord(start);
	  
	  MyKernel<<<gridSize,blockSize>>>(d.data, a.data, b.data, N);
	  
	  cudaEventRecord(stop); 
	  cudaEventSynchronize(stop);
	  float milliseconds = 0;  
	  cudaEventElapsedTime(&milliseconds, start, stop);
	  cudaEventDestroy(start);
	  cudaEventDestroy(stop);
	  cout << milliseconds << " ms" << endl;
	  if(i > 0)
	    average += milliseconds;
	}
      average /= (iterations - 1);
      cout << "Average = " << average << endl;
    }
}
