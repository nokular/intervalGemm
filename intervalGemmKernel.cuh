#ifndef GEMM_CUH
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

#include <cassert>
#include <iostream>
#include <random>

#include "interval.hpp"
template <typename T>
__global__ void naivegemm(interval::Interval<T> *dA,
                          interval::Interval<T> *dB,
                          interval::Interval<T> *dC, const uint M,
                          const uint N, const uint K) {
  const uint i = blockIdx.y * blockDim.y + threadIdx.y;
  const uint j = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < M && j < N) {
    interval::Interval<T> sum = {T(0.0), T(0.0)};
    for (uint k = 0; k < K; k++) {
      const interval::Interval<T> a = dA[i * K + k];
      const interval::Interval<T> b = dB[k * N + j];
      sum = interval::add(sum, interval::mul(a, b));
    }
    dC[i * N + j] = sum;
  }
}

template <typename T, uint TN, uint TM,
          uint WITERN, uint WITERM, uint KTILE, uint BLOCKDIMX, uint BLOCKDIMY,
          uint WTILEN, uint WTILEM, uint BTILEM, uint BTILEN>
__global__ void Gemm(interval::Interval<T> *dA, interval::Interval<T> *dB,
                        interval::Interval<T> *dC, const uint M, const uint N,
                        const uint K) {
  __shared__ interval::Interval<T> As[KTILE][BTILEM + 1];
  __shared__ interval::Interval<T> Bs[KTILE][BTILEN + 1];
  int warpRow = (threadIdx.x / 4) % 8;
  int warpCol = threadIdx.x % 4;
  int warpIdx = threadIdx.x / 32;
  int warpPosIdxX = warpIdx % (BTILEN / WTILEN);
  int warpPosIdxY = warpIdx / (BTILEN / WTILEN);
  interval::Interval<T> regM[WITERM * TM];
  interval::Interval<T> regN[WITERN * TN];
  interval::Interval<T> results[TM * TN * WITERM * WITERN];

// set registers to zero

  for (int i = 0; i < WITERN * TN; i++) {
    regN[i] = interval::Interval<T>{0, 0};
  }

  for (int i = 0; i < WITERM * TM; i++) {
    regM[i] = interval::Interval<T>{0, 0};
  }

// Initialize results

  for (int i = 0; i < WITERM; i++) {

    for (int j = 0; j < WITERN; j++) {

      for (int m = 0; m < TM; m++) {

        for (int n = 0; n < TN; n++) {
          int column = blockIdx.x * BTILEN +
                       (warpIdx) % (BTILEN / WTILEN) * WTILEN + (warpCol)*TN +
                       n + j * TN * 4;
          int row = blockIdx.y * BTILEM +
                    (warpIdx) / (BTILEN / WTILEN) * WTILEM + (warpRow)*TM + m +
                    i * TM * 8;
          if (column < N && row < M) {
            results[i * WITERN * TM * TN + j * TM * TN + m * TN + n] =
                interval::Interval<T>{0, 0};
          } else {
            results[i * WITERN * TM * TN + j * TM * TN + m * TN + n] =
                interval::Interval<T>{0, 0};
          }
        }
      }
    }
  }


  for (int outerDotIdx = 0; outerDotIdx < K; outerDotIdx += KTILE) {
    // load data from A to shared memory
    __syncthreads();
    const int loadingStepsA = KTILE * WITERM * TM / (1 * BLOCKDIMX);
    const int rowsPerBlockStepA = 1 * BLOCKDIMX * BLOCKDIMY / KTILE;

    for (int step = 0; step < loadingStepsA; step++) {
      int noffset = threadIdx.x * 1 % KTILE + outerDotIdx;
      int moffset = (threadIdx.x * 1 / KTILE) + rowsPerBlockStepA * step +
                    blockIdx.y * BTILEM;

      int n_shared = (threadIdx.x * 1) % KTILE;
      int m_shared = (threadIdx.x * 1) / KTILE + step * rowsPerBlockStepA;
      if (moffset < M && noffset < K) {
        As[n_shared][m_shared] = dA[moffset * K + noffset];
      } else {
        As[n_shared][m_shared] = interval::Interval<T>{0, 0};
      }
    }
    const int loadingStepsB = KTILE * WITERN * TN / (1 * BLOCKDIMY);
    const int rowsPerBlockStepB = 1 * BLOCKDIMY * BLOCKDIMX / BTILEN;

    for (int step = 0; step < loadingStepsB; step++) {
      int noffset = blockIdx.x * BTILEN + threadIdx.x * 1 % BTILEN;
      int moffset = outerDotIdx + (threadIdx.x * 1 / BTILEN) +
                    rowsPerBlockStepB * step;
      int n_shared = (threadIdx.x) % BTILEN;
      int m_shared = (threadIdx.x) / BTILEN + step * rowsPerBlockStepB;
      if (moffset < K && noffset < N) {
        Bs[m_shared][n_shared] = dB[moffset * N + noffset];
      } else {
        Bs[m_shared][n_shared] = interval::Interval<T>{0, 0};
      }
    }
    __syncthreads();

    for (int innerDotIdx = 0; innerDotIdx < KTILE; innerDotIdx++) {

      for (int quadrant = 0; quadrant < WITERM; quadrant++) {

        for (int i = 0; i < TM; i++) {
          regM[quadrant * TM + i] =
              As[innerDotIdx][(warpRow * TM) + (warpPosIdxY * WTILEM) + i +
                              quadrant * TM * 8];
        }
      }

      for (int quadrant = 0; quadrant < WITERN; quadrant++) {

        for (int i = 0; i < TN; i++) {
          regN[quadrant * TN + i] =
              Bs[innerDotIdx][(warpCol * TN) + (warpPosIdxX * WTILEN) + i +
                              quadrant * TN * 4];
        }
      }
// warp level computation

      for (int i = 0; i < WITERM; i++) {

        for (int j = 0; j < WITERN; j++) {

          for (int m = 0; m < TM; m++) {

            for (int n = 0; n < TN; n++) {
              // out of bounds check
              results[i * WITERN * TM * TN + j * TM * TN + m * TN + n] =
                  interval::add(
                      results[i * WITERN * TM * TN + j * TM * TN + m * TN + n],
                      interval::mul(regM[i * TM + m], regN[j * TN + n]));
            }
          }
        }
      }
    }
  }
// write results to global memory

  for (int i = 0; i < WITERM; i++) {

    for (int j = 0; j < WITERN; j++) {

      for (int m = 0; m < TM; m++) {

        for (int n = 0; n < TN; n++) {
          int column = blockIdx.x * BTILEN +
                       (warpIdx) % (BTILEN / WTILEN) * WTILEN + (warpCol)*TN +
                       n + j * TN * 4;
          int row = blockIdx.y * BTILEM +
                    (warpIdx) / (BTILEN / WTILEN) * WTILEM + (warpRow)*TM + m +
                    i * TM * 8;
          if (column < N && row < M) {
            dC[row * N + column] =
                results[i * WITERN * TM * TN + j * TM * TN + m * TN + n];
          }
        }
      }
    }
  }
}
template <typename T, uint TN, uint TM,
          uint WITERN, uint WITERM, uint KTILE, uint BLOCKDIMX, uint BLOCKDIMY,
          uint WTILEN, uint WTILEM, uint BTILEM, uint BTILEN,uint shared_memory_size>
void IntervalGemm(const interval::IntervalMatrix<T> &A,
                               const interval::IntervalMatrix<T> &B,
                               interval::IntervalMatrix<T> &C) {
  static_assert(KTILE * WITERN * TN / (BLOCKDIMY * 1) >=1);  // makes sure that always a full row can be loaded into shared memory per loading step
  static_assert(KTILE * WITERM * TM / (BLOCKDIMX * 1) >=1);  // makes sure that always a full row can be loaded into shared memory per loading step
  static_assert(BLOCKDIMX % 4 == 0);  // warp size is 32
  static_assert(BLOCKDIMY % 8 == 0);  // warp size is 32
  static_assert((BLOCKDIMY * KTILE * WITERM * TM + BLOCKDIMX * KTILE * WITERN * TN) * sizeof(interval::Interval<T>) <= shared_memory_size);
  assert(A.cols == B.rows);
  const uint M = A.rows;  // Height of matrix A
  const uint N = B.cols;  // Width of matrix B
  const uint K = A.cols;  // Width of matrix A and height of matrix B
  interval::Interval<T> *dA, *dB, *dC;

  cudaMalloc(&dA, A.rows * A.cols * sizeof(interval::Interval<T>));
  cudaMalloc(&dB, B.rows * B.cols * sizeof(interval::Interval<T>));
  cudaMalloc(&dC, C.rows * C.cols * sizeof(interval::Interval<T>));

  // copy data to device
  cudaMemcpy(dA, A.data, A.rows * A.cols * sizeof(interval::Interval<T>),
             cudaMemcpyHostToDevice);
  cudaMemcpy(dB, B.data, B.rows * B.cols * sizeof(interval::Interval<T>),
             cudaMemcpyHostToDevice);
  cudaMemcpy(dC, C.data, C.rows * C.cols * sizeof(interval::Interval<T>),
             cudaMemcpyHostToDevice);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    exit(1);
  }
  dim3 grid;
  dim3 block;

  block = dim3(BLOCKDIMX * BLOCKDIMY);
  grid = dim3((N + BTILEN - 1) / BTILEN, (M + BTILEM - 1) / (BTILEM));
  // start timing
  Gemm<T, TN, TM,
          WITERN, WITERM, KTILE, BLOCKDIMX, BLOCKDIMY, WTILEN, WTILEM,
          BTILEM, BTILEN><<<grid, block>>>(dA, dB, dC, M, N, K);
  // stop timing
  
  // copy result back to host
  cudaMemcpy(C.data, dC, C.rows * C.cols * sizeof(interval::Interval<T>),
             cudaMemcpyDeviceToHost);
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    exit(1);
  }

  // free allocated memory
  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dC);
}
#endif