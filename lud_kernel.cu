#include <cuda.h>
#include <stdio.h>
#include <cusolverDn.h>

#include "common.h"

#ifdef RD_WG_SIZE_0_0
        #define BLOCK_SIZE RD_WG_SIZE_0_0
#elif defined(RD_WG_SIZE_0)
        #define BLOCK_SIZE RD_WG_SIZE_0
#elif defined(RD_WG_SIZE)
        #define BLOCK_SIZE RD_WG_SIZE
#else
        #define BLOCK_SIZE 16
#endif


__global__ void 
lud_diagonal(float *m, int matrix_dim, int offset)
{
  int i,j;
  __shared__ float shadow[BLOCK_SIZE][BLOCK_SIZE];

  int array_offset = offset*matrix_dim+offset;
  for(i=0; i < BLOCK_SIZE; i++){
    shadow[i][threadIdx.x]=m[array_offset+threadIdx.x];
    array_offset += matrix_dim;
  }
  __syncthreads();
  for(i=0; i < BLOCK_SIZE-1; i++) {

    if (threadIdx.x>i){
      for(j=0; j < i; j++)
        shadow[threadIdx.x][i] -= shadow[threadIdx.x][j]*shadow[j][i];
      shadow[threadIdx.x][i] /= shadow[i][i];
    }

    __syncthreads();
    if (threadIdx.x>i){

      for(j=0; j < i+1; j++)
        shadow[i+1][threadIdx.x] -= shadow[i+1][j]*shadow[j][threadIdx.x];
    }
    __syncthreads();
  }

  /* 
     The first row is not modified, it
     is no need to write it back to the
     global memory

   */
  array_offset = (offset+1)*matrix_dim+offset;
  for(i=1; i < BLOCK_SIZE; i++){
    m[array_offset+threadIdx.x]=shadow[i][threadIdx.x];
    array_offset += matrix_dim;
  }
}

__global__ void
lud_perimeter(float *m, int matrix_dim, int offset)
{
  __shared__ float dia[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ float peri_row[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ float peri_col[BLOCK_SIZE][BLOCK_SIZE];

  int i,j, array_offset;
  int idx;

  if (threadIdx.x < BLOCK_SIZE) {
    idx = threadIdx.x;
    
    array_offset = offset*matrix_dim+offset;
    for (i=0; i < BLOCK_SIZE/2; i++){
      dia[i][idx]=m[array_offset+idx];
      array_offset += matrix_dim;
    }
    
    array_offset = offset*matrix_dim+offset;
    for (i=0; i < BLOCK_SIZE; i++) {
      peri_row[i][idx]=m[array_offset+(blockIdx.x+1)*BLOCK_SIZE+idx];
      array_offset += matrix_dim;
    }

  } else {
    idx = threadIdx.x-BLOCK_SIZE;
    
    array_offset = (offset+BLOCK_SIZE/2)*matrix_dim+offset;
    for (i=BLOCK_SIZE/2; i < BLOCK_SIZE; i++){
      dia[i][idx]=m[array_offset+idx];
      array_offset += matrix_dim;
    }
    
    array_offset = (offset+(blockIdx.x+1)*BLOCK_SIZE)*matrix_dim+offset;
    for (i=0; i < BLOCK_SIZE; i++) {
      peri_col[i][idx] = m[array_offset+idx];
      array_offset += matrix_dim;
    }
  
  }
  __syncthreads();

/* this version works ok on hardware, but not gpgpusim
 **************************************************************
  if (threadIdx.x < BLOCK_SIZE) { //peri-row
    idx=threadIdx.x;
    for(i=1; i < BLOCK_SIZE; i++){
      for (j=0; j < i; j++)
        peri_row[i][idx]-=dia[i][j]*peri_row[j][idx];
    }

    
    array_offset = (offset+1)*matrix_dim+offset;
    for(i=1; i < BLOCK_SIZE; i++){
      m[array_offset+(blockIdx.x+1)*BLOCK_SIZE+idx] = peri_row[i][idx];
      array_offset += matrix_dim;
    }
  } else { //peri-col
    idx=threadIdx.x - BLOCK_SIZE;
    for(i=0; i < BLOCK_SIZE; i++){
      for(j=0; j < i; j++)
        peri_col[idx][i]-=peri_col[idx][j]*dia[j][i];
      peri_col[idx][i] /= dia[i][i];
    }

    __syncthreads();
    
    array_offset = (offset+(blockIdx.x+1)*BLOCK_SIZE)*matrix_dim+offset;
    for(i=0; i < BLOCK_SIZE; i++){
      m[array_offset+idx] =  peri_col[i][idx];
      array_offset += matrix_dim;
    }
  }
***************************************************************/
  if (threadIdx.x < BLOCK_SIZE) { //peri-row
    idx=threadIdx.x;
    for(i=1; i < BLOCK_SIZE; i++){
      for (j=0; j < i; j++)
        peri_row[i][idx]-=dia[i][j]*peri_row[j][idx];
    }
  } else { //peri-col
    idx=threadIdx.x - BLOCK_SIZE;
    for(i=0; i < BLOCK_SIZE; i++){
      for(j=0; j < i; j++)
        peri_col[idx][i]-=peri_col[idx][j]*dia[j][i];
      peri_col[idx][i] /= dia[i][i];
    }
  }

  __syncthreads();
    
  if (threadIdx.x < BLOCK_SIZE) { //peri-row
    idx=threadIdx.x;
    array_offset = (offset+1)*matrix_dim+offset;
    for(i=1; i < BLOCK_SIZE; i++){
      m[array_offset+(blockIdx.x+1)*BLOCK_SIZE+idx] = peri_row[i][idx];
      array_offset += matrix_dim;
    }
  } else { //peri-col
    idx=threadIdx.x - BLOCK_SIZE;
    array_offset = (offset+(blockIdx.x+1)*BLOCK_SIZE)*matrix_dim+offset;
    for(i=0; i < BLOCK_SIZE; i++){
      m[array_offset+idx] =  peri_col[i][idx];
      array_offset += matrix_dim;
    }
  }

}

__global__ void
lud_internal(float *m, int matrix_dim, int offset)
{
  __shared__ float peri_row[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ float peri_col[BLOCK_SIZE][BLOCK_SIZE];

  int i;
  float sum;

  int global_row_id = offset + (blockIdx.y+1)*BLOCK_SIZE;
  int global_col_id = offset + (blockIdx.x+1)*BLOCK_SIZE;

  peri_row[threadIdx.y][threadIdx.x] = m[(offset+threadIdx.y)*matrix_dim+global_col_id+threadIdx.x];
  peri_col[threadIdx.y][threadIdx.x] = m[(global_row_id+threadIdx.y)*matrix_dim+offset+threadIdx.x];

  __syncthreads();

  sum = 0;
  for (i=0; i < BLOCK_SIZE; i++)
    sum += peri_col[threadIdx.y][i] * peri_row[i][threadIdx.x];
  m[(global_row_id+threadIdx.y)*matrix_dim+global_col_id+threadIdx.x] -= sum;


}

// m is a device copy of the matrix, it's NOT a pointer to host memory
void lud_cuda(float *m, int matrix_dim)
{
  int i=0;
  dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
  float *m_debug = (float*)malloc(matrix_dim*matrix_dim*sizeof(float));

  for (i=0; i < matrix_dim-BLOCK_SIZE; i += BLOCK_SIZE) {
      lud_diagonal<<<1, BLOCK_SIZE>>>(m, matrix_dim, i);
      lud_perimeter<<<(matrix_dim-i)/BLOCK_SIZE-1, BLOCK_SIZE*2>>>(m, matrix_dim, i);
      dim3 dimGrid((matrix_dim-i)/BLOCK_SIZE-1, (matrix_dim-i)/BLOCK_SIZE-1);
      lud_internal<<<dimGrid, dimBlock>>>(m, matrix_dim, i); 
  }
  lud_diagonal<<<1,BLOCK_SIZE>>>(m, matrix_dim, i);
}

void lud_cusolver(float *device_m, int matrix_dim)
{
  int m=matrix_dim, n=matrix_dim, ld = matrix_dim;
  size_t matrix_size = sizeof(float)*matrix_dim*matrix_dim;
  float *deviceC, *hostC;
  cudaMalloc(&deviceC, matrix_size);
  cudaMemcpy(deviceC, hostC, matrix_size, cudaMemcpyHostToDevice);


  cublasHandle_t cublasHandle;
  cublasCreate(&cublasHandle); // note we dont catch error here!
  
  // Matrix transposition using cuBLAS Sgeam see documentation at:
  // https://docs.nvidia.com/cuda/cublas/index.html#cublas-t-geam
  cublasOperation_t transa = CUBLAS_OP_T;
  cublasOperation_t transb = CUBLAS_OP_N;
  float alpha=1, beta=0;
  cublasStatus_t blasStatus = cublasSgeam(cublasHandle, transa, transb,
                                  m, n,
                                  &alpha,
                                  device_m, ld,
                                  &beta,
                                  nullptr, ld,
                                  deviceC, ld);
    
  
  cusolverDnHandle_t cusolverHandle;
  cusolverStatus_t cusolverStatus = cusolverDnCreate(&cusolverHandle);
  cusolverDnParams_t cusolverParams;
  cusolverDnCreateParams(&cusolverParams); // default initialization we don't use advanced options

  // Creating Host and Device Buffers Required by LU solver
  size_t hostBufferSize = 0, deviceBufferSize=0;
  cusolverDnXgetrf_bufferSize(cusolverHandle, cusolverParams,
    m, n,
    CUDA_R_32F, deviceC, ld,
    CUDA_R_32F,
    &deviceBufferSize,
    &hostBufferSize);
  float *deviceBuffer, *hostBuffer;
  cudaMalloc(&deviceBuffer, sizeof(float)*deviceBufferSize);
  hostBuffer = (float *) malloc(sizeof(float)*deviceBufferSize);


  //LU decomposition using cuSOLVER cusolverDnXgetrf()
  // see docoumentation at: https://docs.nvidia.com/cuda/cusolver/index.html#cusolverdnxgetrf
  int hostInfo=0;
  int* deviceInfo;
  cudaMemcpy(deviceInfo, &hostInfo, sizeof(int), cudaMemcpyHostToDevice);
  cusolverDnXgetrf(cusolverHandle, cusolverParams,
      m, n, CUDA_R_32F, deviceC, ld, nullptr, CUDA_R_32F,
      deviceBuffer, deviceBufferSize, hostBuffer, hostBufferSize, deviceInfo);

  cudaMemcpy(&hostInfo, deviceInfo, sizeof(int), cudaMemcpyDeviceToHost);
  printf("info (should be 0 if LU successful) %d\n", hostInfo);

  // Transpose matrix back
  cublasSgeam(cublasHandle, transa, transb,
    m, n,
    &alpha,
    deviceC, ld,
    &beta,
    nullptr, ld,
    device_m, ld);
    
  cudaMemcpy(hostC, device_m, matrix_size, cudaMemcpyDeviceToHost);
  //printf("LU decomposed A matrix print:\n");
  //print_matrix(hostC, matrix_dim);

  // Destroy cuBLAS Handle
  cublasDestroy(cublasHandle);
  // Destroy cusolverDnParams
  cusolverDnDestroyParams(cusolverParams);
  // Destroy cuSOLVER Handle
  cusolverDnDestroy(cusolverHandle);

  cudaFree(deviceC);

}


void lud_cusolver_streaming(float *device_m, int matrix_dim, cudaStream_t& stream)
{
  int m=matrix_dim, n=matrix_dim, ld = matrix_dim;
  size_t matrix_size = sizeof(float)*matrix_dim*matrix_dim;
  float *deviceC, *hostC;
  cudaMalloc(&deviceC, matrix_size);
  cudaMemcpyAsync(deviceC, hostC, matrix_size, cudaMemcpyHostToDevice, stream);


  cublasHandle_t cublasHandle;
  cublasCreate(&cublasHandle); // note we dont catch error here!
  cublasSetStream(cublasHandle, stream); 


  // Matrix transposition using cuBLAS Sgeam see documentation at:
  // https://docs.nvidia.com/cuda/cublas/index.html#cublas-t-geam
  cublasOperation_t transa = CUBLAS_OP_T;
  cublasOperation_t transb = CUBLAS_OP_N;
  float alpha=1, beta=0;
  cublasStatus_t blasStatus = cublasSgeam(cublasHandle, transa, transb,
                                  m, n,
                                  &alpha,
                                  device_m, ld,
                                  &beta,
                                  nullptr, ld,
                                  deviceC, ld);
    
  
  cusolverDnHandle_t cusolverHandle;
  cusolverStatus_t cusolverStatus = cusolverDnCreate(&cusolverHandle);
  cusolverStatus = cusolverDnSetStream(cusolverHandle, stream);
  cusolverDnParams_t cusolverParams;
  cusolverDnCreateParams(&cusolverParams); // default initialization we don't use advanced options

  // Creating Host and Device Buffers Required by LU solver
  size_t hostBufferSize = 0, deviceBufferSize=0;
  cusolverDnXgetrf_bufferSize(cusolverHandle, cusolverParams,
    m, n,
    CUDA_R_32F, deviceC, ld,
    CUDA_R_32F,
    &deviceBufferSize,
    &hostBufferSize);
  float *deviceBuffer, *hostBuffer;
  cudaMalloc(&deviceBuffer, sizeof(float)*deviceBufferSize);
  hostBuffer = (float *) malloc(sizeof(float)*deviceBufferSize);


  //LU decomposition using cuSOLVER cusolverDnXgetrf()
  // see docoumentation at: https://docs.nvidia.com/cuda/cusolver/index.html#cusolverdnxgetrf
  int hostInfo=0;
  int* deviceInfo;
  cudaMemcpyAsync(deviceInfo, &hostInfo, sizeof(int), cudaMemcpyHostToDevice, stream);
  cusolverDnXgetrf(cusolverHandle, cusolverParams,
      m, n, CUDA_R_32F, deviceC, ld, nullptr, CUDA_R_32F,
      deviceBuffer, deviceBufferSize, hostBuffer, hostBufferSize, deviceInfo);
  cudaMemcpyAsync(&hostInfo, deviceInfo, sizeof(int), cudaMemcpyDeviceToHost, stream);
#ifdef DEBUG
  printf("info (should be 0 if LU successful) %d\n", hostInfo);
#endif
  // Transpose matrix back
  cublasSgeam(cublasHandle, transa, transb,
    m, n,
    &alpha,
    deviceC, ld,
    &beta,
    nullptr, ld,
    device_m, ld);
    
  cudaMemcpyAsync(hostC, device_m, matrix_size, cudaMemcpyDeviceToHost, stream);
  //printf("LU decomposed A matrix print:\n");
  //print_matrix(hostC, matrix_dim);

  // Destroy cuBLAS Handle
  cublasDestroy(cublasHandle);
  // Destroy cusolverDnParams
  cusolverDnDestroyParams(cusolverParams);
  // Destroy cuSOLVER Handle
  cusolverDnDestroy(cusolverHandle);

  cudaFree(deviceC);

}


