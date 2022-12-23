

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
  size_t sizeA = sizeof(data_type)*N*N;
  size_t sizeC = sizeA;
  float *deviceA, *deviceC;
  cudaMalloc(&deviceA, sizeA);
  cudaMalloc(&deviceC, sizeC);
  
  cudaMemcpy(deviceA, hostA, sizeA, cudaMemcpyHostToDevice);
  cudaMemcpy(deviceC, hostC, sizeC, cudaMemcpyHostToDevice);

  std::printf("A matrix print:\n");
  print_matrix(hostA, N); 

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
                                  deviceA, lda,
                                  &beta,
                                  nullptr, ldb,
                                  deviceC, ldc);
    
  

  cudaMemcpy(hostC, deviceC, sizeC, cudaMemcpyDeviceToHost);
  
  std::printf("A matrix print:\n");
  print_matrix(hostA, N);
  std::printf("C matrix print:\n");
  print_matrix(hostC, N);
  
  

  cusolverDnHandle_t cusolverHandle;
  cusolverStatus_t cusolverStatus = cusolverDnCreate(&cusolverHandle);
  cusolverDnParams_t cusolverParams;
  cusolverDnCreateParams(&cusolverParams); // default initialization we don't use advanced options

  // Creating Host and Device Buffers Required by LU solver
  size_t hostBufferSize = 0, deviceBufferSize=0;
  cusolverDnXgetrf_bufferSize(cusolverHandle, cusolverParams,
    m, n,
    CUDA_R_32F, deviceC, ldc,
    CUDA_R_32F,
    &deviceBufferSize,
    &hostBufferSize);
  data_type *deviceBuffer, *hostBuffer;
  cudaMalloc(&deviceBuffer, sizeof(data_type)*deviceBufferSize);
  hostBuffer = (data_type *) malloc(sizeof(data_type)*deviceBufferSize);


  //LU decomposition using cuSOLVER cusolverDnXgetrf()
  // see docoumentation at: https://docs.nvidia.com/cuda/cusolver/index.html#cusolverdnxgetrf
  int hostInfo=0;
  int* deviceInfo;
  cudaMemcpy(deviceInfo, &hostInfo, sizeof(int), cudaMemcpyHostToDevice);
  cusolverDnXgetrf(cusolverHandle, cusolverParams,
      m, n, CUDA_R_32F, deviceC, ldc, nullptr, CUDA_R_32F,
      deviceBuffer, deviceBufferSize, hostBuffer, hostBufferSize, deviceInfo);

  cudaMemcpy(&hostInfo, deviceInfo, sizeof(int), cudaMemcpyDeviceToHost);
  printf("info (should be 0 if LU successful) %d\n", hostInfo);


  cudaMemcpy(hostC, deviceC, sizeC, cudaMemcpyDeviceToHost);
  std::printf("LU decomposed C matrix print:\n");
  print_matrix(hostC, N);

  cublasSgeam(cublasHandle, transa, transb,
    m, n,
    &alpha,
    deviceC, ldc,
    &beta,
    nullptr, ldb,
    deviceA, lda);
    
  cudaMemcpy(hostA, deviceA, sizeA, cudaMemcpyDeviceToHost);
  std::printf("LU decomposed A matrix print:\n");
  print_matrix(hostA, N);

  // Destroy cuBLAS Handle
  cublasDestroy(cublasHandle);
  // Destroy cusolverDnParams
  cusolverDnDestroyParams(cusolverParams);
  // Destroy cuSOLVER Handle
  cusolverDnDestroy(cusolverHandle);


  cudaFree(deviceA);
  cudaFree(deviceC);

}
