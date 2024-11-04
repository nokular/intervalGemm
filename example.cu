#include <stdio.h>

#include <iostream>
// include a timing library
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <random>
#include <vector>

#include "intervalGemmKernel.cuh"
#include "interval.hpp"

int32_t floatToInt(float f) {
  int32_t intRep;
  memcpy(&intRep, &f, sizeof(f));
  if (intRep < 0) {
    intRep = (1 << 31) - intRep;
  }
  return intRep;
}

// Compute ULP difference between two floats
int32_t ulpDifference(float a, float b) {
  int32_t intA = floatToInt(a);
  int32_t intB = floatToInt(b);
  return std::abs(intA - intB);
}
int main(int argc, char **argv) {
  int deviceCount;
  cudaGetDeviceCount(&deviceCount);
  if (deviceCount == 0) {
    std::cerr << "No CUDA device found" << std::endl;
    return 1;
  }

  // initialize random generator
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dis(0.0, 1.0);
  float a = 1.3213 + 20;
  float b = 1.3213 + 20;
  float c = 0;

  // create 2 interval matrices with random values of size 2x2
  int dimM = 2048;
  int dimN = 2048;
  int dimK = 2048;

  const int TN = 2;
  const int TM = 2;
  const int WITERN = 2;
  const int WITERM = 2;
  const int KTILE = 16;
  const int BLOCKDIMX = 16;
  const int BLOCKDIMY = 16;
  const int WTILEN = TN * WITERN * 4;
  const int WTILEM = TM * WITERM * 8;
  const int BTILEM = WITERM * TM * BLOCKDIMY;
  const int BTILEN = WITERN * TN * BLOCKDIMX;
  const int SHAREDMEMORYSIZE = 48000;
  interval::IntervalMatrix<float> A =
      interval::InitHostMatrix<float>(dimM, dimK);
  interval::IntervalMatrix<float> B =
      interval::InitHostMatrix<float>(dimK, dimN);
  interval::IntervalMatrix<float> C =
      interval::InitHostMatrix<float>(dimM, dimN);

  float *hA = new float[dimM * dimK];
  float *hB = new float[dimK * dimN];
  float *hC = new float[dimM * dimN];
  for (int i = 0; i < dimM * dimK; i++) {
    A.data[i].low = a;
    A.data[i].high = a;
    hA[i] = a;
  }
  for (int i = 0; i < dimK * dimN; i++) {
    B.data[i].low = b;
    B.data[i].high = b;
    hB[i] = b;
  }
  for (int i = 0; i < dimM * dimN; i++) {
    C.data[i].low = c;
    C.data[i].high = c;
    hC[i] = c;
  }
  IntervalGemm<float,TN, TM, WITERN, WITERM,
        KTILE, BLOCKDIMX, BLOCKDIMY, WTILEN, WTILEM, BTILEM, BTILEN,SHAREDMEMORYSIZE>(A, B, C);
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl;
    return 1;
  }

  int32_t total_ulps = 0;
  float total_diff = 0.0f;
  int t = 0;
  for (int i = 0; i < dimM; ++i) {
    for (int j = 0; j < dimN; ++j) {
      int32_t ulps =
          ulpDifference(C.data[i * dimN + j].low, C.data[i * dimN + j].high);
      total_ulps += ulps;
      total_diff +=
          std::abs(C.data[i * dimN + j].low - C.data[i * dimN + j].high);
    }
  }

  total_diff /= dimM * dimN;
  std::cout << "Avg diff: " << total_diff << std::endl;
  std::cout << t << std::endl;
  std::cout << "Block: " << BLOCKDIMX << " " << BLOCKDIMY << std::endl;
  // dim3((N + BTILEN - 1) / BTILEN, (M + BTILEM - 1) / (BTILEM));
  std::cout << "Grid: " << (dimN + BTILEN - 1) / BTILEN << " "
            << (dimM + BTILEM - 1) / BTILEM << std::endl;
  uint m = A.rows;  // Height of matrix A
  uint n = B.cols;  // Width of matrix B
  uint k = A.cols;  //
  // print all the TM, TN, WITERN, WITERM, 1, KTILE, BLOCKDIMX, BLOCKDIMY,
  // WTILEN, WTILEM, BTILEM, BTILEN
  std::cout << "TM: " << TM << " TN: " << TN << " WITERN: " << WITERN
            << " WITERM: " << WITERM << " KTILE: " << KTILE
            << " BLOCKDIMX: " << BLOCKDIMX << " BLOCKDIMY: " << BLOCKDIMY
            << " WTILEN: " << WTILEN << " WTILEM: " << WTILEM
            << " BTILEM: " << BTILEM << " BTILEN: " << BTILEN << std::endl;
  // print N,m,k
  std::cout << "N: " << n << " M: " << m << " K: " << k << std::endl;
  std::cout << "Total ULPs: " << total_ulps << std::endl;
  std::cout << "Average ULP difference: " << total_ulps / (dimM * dimN)
            << std::endl;
  std::cout << "single" << std::endl;
  // free allocated memory
  delete[] A.data;
  delete[] B.data;
  delete[] C.data;
  delete[] hA;
  delete[] hB;
  delete[] hC;

  return 0;
}