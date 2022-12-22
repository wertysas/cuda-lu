

#include <cstdio>

#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <cublas_v2.h>

#include "common.h"

#define N 3

using data_type = float;


int main() {
  float hostA[9] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  float hostC[9];
  int  m=N, n=N, lda=N, ldb=N, ldc=N;
  float alpha=1, beta=0;
  size_t sizeA = sizeof(data_type)*N*N;
  size_t sizeC = sizeA;
  float *deviceA, *deviceC;
  cudaMalloc(&deviceA, sizeA);
  cudaMalloc(&deviceC, sizeC);
  
  cudaMemcpy(deviceA, hostA, sizeA, cudaMemcpyHostToDevice);
  cudaMemcpy(deviceC, hostC, sizeC, cudaMemcpyHostToDevice);
    
  cublasHandle_t* cublasH = nullptr;
  cublasCreate(cublasH); // note we dont catch error here!
  // cudaStream_t stream;
  // cudaStreamCreate(&stream);
  // cublasSetStream(*cublasH, stream); 

  cublasOperation_t transa = CUBLAS_OP_T;
  cublasOperation_t transb = CUBLAS_OP_T;
  

  cublasStatus_t err = cublasSgeam(*cublasH, transa, transb,
                                  m, n,
                                  &alpha,
                                  deviceA, lda,
                                  &beta,
                                  nullptr, ldb,
                                  deviceC, ldc);
    
  

  cudaMemcpy(hostC, deviceC, sizeC, cudaMemcpyDeviceToHost);
  
  std::printf("A matrix print:\n");
  print_matrix(hostA, N);
  std::printf("C matrix print:\n");
  print_matrix(hostC, N);
  // cusolverDnHandle_t* cusolverHandle;
  // cusolverStatus_t cusolverStatus = cusolverDnCreate(cusolverHandle);



}


