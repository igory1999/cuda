#include <iostream>
#include <random>

using namespace std;


// Matrices are stored in row-major order:
// M(row, column) = *(M.elements + row * M.width + col)
typedef struct
{
  int width;
  int height;
  float * elements;
} Matrix;


// Thread block size
#define BLOCK_SIZE 16

// Forward declaration of the matrix multiplication kernel
__global__ void MatMulKernel(const Matrix, const Matrix, Matrix);

// Forward declaration of sequential CPU function
void sequential_cpu(Matrix A, Matrix B, Matrix C);

// Matrix multiplication - Host code
// Matrix dimensions are assumed to be multiples of BLOCK_SIZE
void MatMu(const Matrix A, const Matrix B, Matrix C)
{
  // Load A and B to device memory
  Matrix d_A;
  d_A.width = A.width;
  d_A.height = A.height;
  size_t size = A.width * A.height * sizeof(float);
  cudaMalloc(&d_A.elements, size);
  cudaMemcpy(d_A.elements, A.elements, size, cudaMemcpyHostToDevice);

  Matrix d_B;
  d_B.width = B.width;
  d_B.height = B.height;
  size = B.width * B.height * sizeof(float);
  cudaMalloc(&d_B.elements, size);
  cudaMemcpy(d_B.elements, B.elements, size, cudaMemcpyHostToDevice);  

  // Allocate C in device memory
  Matrix d_C;
  d_C.width = C.width;
  d_C.height = C.height;
  size = C.width * C.height * sizeof(float);
  cudaMalloc(&d_C.elements, size);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // Invoke kernel
  dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
  dim3 dimGrid(B.width/dimBlock.x, A.height/dimBlock.y);

  cudaEventRecord(start);
  MatMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);
  cudaEventRecord(stop);
  // Read C from device memory
  cudaMemcpy(C.elements, d_C.elements, size, cudaMemcpyDeviceToHost);
  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);

  cout << "Kernel call took " << milliseconds << " milliseconds" << endl;

  // Free device memory
  cudaFree(d_A.elements);
  cudaFree(d_B.elements);
  cudaFree(d_C.elements);
  /*
  cudaEventRecord(start);
  sequential_cpu(A, B, C);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milliseconds, start, stop);

  cout << "Sequential CPU function call took " << milliseconds << " milliseconds" << endl;
  */
}

// Matrix multiplication kernel called by MatMul()
__global__ void MatMulKernel(Matrix A, Matrix B, Matrix C)
{
  // Each thread computes one element of C
  // by accumulating results into Cvalue

  float Cvalue = 0;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  for (int e = 0; e < A.width; ++e)
    Cvalue += A.elements[row * A.width + e] * B.elements[e * B.width + col];
  C.elements[row * C.width + col] = Cvalue;
  
}

// Sequential CPU version is given for comparison
void sequential_cpu(Matrix A, Matrix B, Matrix C)
{
  for(int i = 0; i < C.height; ++i)
    {
      for(int j = 0; j < C.width; ++j)
	{
	  C.elements[i*C.width + j] = 0;
	  for(int ac = 0; ac < A.width; ++ac)
	    {
	      for(int br = 0; br < B.height; ++br)
		{
		  C.elements[i*C.width + j] += A.elements[i*A.width + ac]*B.elements[j + br*B.width];
		}
	    }
	}
    }
}

int main()
{
  int n;
  size_t size;

  std::default_random_engine generator;
  std::uniform_real_distribution<float> distribution(-1.0,1.0);
  
  Matrix A;
  A.width = BLOCK_SIZE*150;
  A.height = BLOCK_SIZE*100;
  n = A.width * A.height;
  size = n * sizeof(float);
  A.elements = (float*)malloc(size);
  for(int i = 0; i < n; ++i)
    A.elements[i] = distribution(generator);


  Matrix B;
  B.width = BLOCK_SIZE*200;
  B.height = A.width;
  n = B.width * B.height;
  size = n * sizeof(float);
  B.elements = (float*)malloc(size);
  for(int i = 0; i < n; ++i)
    B.elements[i] = distribution(generator);      


  Matrix C;
  C.width = B.width;
  C.height = A.height;
  n = C.width * C.height;
  size = n * sizeof(float);
  C.elements = (float*)malloc(size);

  for(int i = 0; i < 5; ++i)
    {
      printf("i=%d\n",i);
      MatMu(A, B, C);
    }
  
}
