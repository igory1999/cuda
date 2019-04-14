#include <iostream>
#include <random>

using namespace std;


// Matrices are stored in row-major order:
// M(row, column) = *(M.elements + row * M.stride + col)
typedef struct
{
  int width;
  int height;
  int stride;
  float * elements;
} Matrix;

// Thread block size
#define BLOCK_SIZE 16


// Get a matrix element
__device__ float GetElement(const Matrix A, int row, int col)
{
  return A.elements[row * A.stride + col];
}

// Set a matrix element
__device__ void SetElement(Matrix A, int row, int col, float value)
{
  A.elements[row * A.stride + col] = value;
}

// Get the BLOCK_SIZE x BLOCK_SIZE sub-matrix Asub of A that is located 
// col sub-matrices to the right and row sub-matrices down from the
// upper-left corner of A
__device__ Matrix GetSubMatrix(Matrix A, int row, int col)
{
  Matrix Asub;
  Asub.width = BLOCK_SIZE;
  Asub.height = BLOCK_SIZE;
  Asub.stride = A.stride;
  Asub.elements = &A.elements[A.stride * BLOCK_SIZE * row + 
			      BLOCK_SIZE * col];
  return Asub;
}




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
  d_A.width = d_A.stride = A.width;
  d_A.height = A.height;
  size_t size = A.width * A.height * sizeof(float);
  cudaMalloc(&d_A.elements, size);
  cudaMemcpy(d_A.elements, A.elements, size, cudaMemcpyHostToDevice);

  Matrix d_B;
  d_B.width = d_B.stride = B.width;
  d_B.height = B.height;
  size = B.width * B.height * sizeof(float);
  cudaMalloc(&d_B.elements, size);
  cudaMemcpy(d_B.elements, B.elements, size, cudaMemcpyHostToDevice);  

  // Allocate C in device memory
  Matrix d_C;
  d_C.width = d_C.stride = C.width;
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

  // Block row and column

  int blockRow = blockIdx.y;
  int blockCol = blockIdx.x;

  // Each thread block computes one sub-matrix Csub of C
  Matrix Csub = GetSubMatrix(C, blockRow, blockCol);

  // Each thread computes one element of Csub
  // by accumulating results into Cvalue
  float Cvalue = 0;

  int row = threadIdx.y;
  int col = threadIdx.x;

  // Loop over all the sub-matrices of A and B that are required to compute Csub
  // Multiply each pair of sub-matrices together
  // and accumulate the results
  for(int m = 0; m < (A.width/BLOCK_SIZE); ++m)
    {
      // Get sub-matrix Asub of A
      Matrix Asub = GetSubMatrix(A, blockRow, m);
      
      // Get sub-matrix Bsub of B
      Matrix Bsub = GetSubMatrix(B, m, blockCol);

      // Shared memory used to store Asub and Bsub respectively
      __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
      __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

      // Load Asub and Bsub from device memory to shared memory
      // Each thread loads one element of each sub-matrix
      As[row][col] = GetElement(Asub, row, col);
      Bs[row][col] = GetElement(Bsub, row, col);

      // Synchronize to make sure the sub-matrices are loaded
      // before starting the computation

      __syncthreads();
      // Multiply Asub and Bsub together
      for(int e = 0; e < BLOCK_SIZE; ++e)
	{
	  Cvalue += As[row][e] * Bs[e][col];
	}

      // Synchronize to make sure that the preceding
      // computation is done before load two new sub-matrices of A and B in the next
      // iteration
      __syncthreads();
    }
  // Write Csub to device memory
  // Each thread writes one element
  SetElement(Csub, row, col, Cvalue);
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
  A.width = A.stride = BLOCK_SIZE*150;
  A.height = BLOCK_SIZE*100;
  n = A.width * A.height;
  size = n * sizeof(float);
  A.elements = (float*)malloc(size);
  for(int i = 0; i < n; ++i)
    A.elements[i] = distribution(generator);


  Matrix B;
  B.width = B.stride = BLOCK_SIZE*200;
  B.height = A.width;
  n = B.width * B.height;
  size = n * sizeof(float);
  B.elements = (float*)malloc(size);
  for(int i = 0; i < n; ++i)
    B.elements[i] = distribution(generator);      


  Matrix C;
  C.width = C.stride = B.width;
  C.height = A.height;
  n = C.width * C.height;
  size = n * sizeof(float);
  C.elements = (float*)malloc(size);

  for(int i = 0; i < 5; ++i)
    {
      printf("i = %d\n", i);
      MatMu(A, B, C);
    }
  
}
