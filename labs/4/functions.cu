#include <iostream>
#include <random>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <string>

using namespace std;

void init_h_array(float *a, int n)
{
  std::default_random_engine generator;
  std::uniform_real_distribution<float> distribution(-1.0,1.0);

  for(int i = 0; i < n; ++i)
    a[i] = distribution(generator);
} 

__global__ void map(float *d_a, int n, int m)
{
  int id = blockDim.x * blockIdx.x  + threadIdx.x;
  
  if(id < n)
    for(int i = 0; i < m; ++i)
      {
	d_a[id] = sinf(d_a[id]) + cosf(d_a[id]);
	d_a[id] = expf(d_a[id]);
	d_a[id] = rsqrtf(d_a[id]) - d_a[id];
      }
}

void print_sample(float *a, int m, string msg)
{
  cout << msg << endl;
  for(int i = 0; i < m; ++i)
    cout << "a[" << i << "] = " << a[i] << ", ";
  cout << endl;
}

int main()
{
  const int N = 1024*1024*128;
  const int threads_in_block = 1024;
  const int iterations = 15;


  size_t size = N*sizeof(float);
  float *h_a = (float*)malloc(size);
  init_h_array(h_a, N);

  print_sample(h_a, 5, "initial array");

  float *d_a = NULL;
  cudaMalloc((void **)&d_a, size);

  dim3 block_size = dim3(threads_in_block, 1, 1);
  dim3 grid_size = dim3(N/block_size.x, 1, 1);


  for(int i = 0; i < 5; ++i)
    {
      init_h_array(h_a, N);
      cudaEvent_t start, stop;
      cudaEventCreate(&start);
      cudaEventCreate(&stop);

      cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);

      cudaEventRecord(start);
      map<<<grid_size, block_size>>>(d_a, N, iterations);
      cudaEventRecord(stop); 
      cudaEventSynchronize(stop);
      float milliseconds = 0;  
      cudaEventElapsedTime(&milliseconds, start, stop);
      cudaMemcpy(h_a, d_a, size, cudaMemcpyDeviceToHost);

      print_sample(h_a, 5, "final array");
      
      cout << "H2D, kernel, D2H took " << milliseconds << " ms" << endl;

      cudaEventDestroy(start);
      cudaEventDestroy(stop);
    }


  cudaFree(d_a);
  free(h_a);
}
