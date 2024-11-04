#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <stdint.h>
#include <vector>
#include <fstream>
#include <cuda.h>
#include <cuda_runtime.h>
#ifndef INTERVAL_HPP
#define INTERVAL_HPP

#define TILE_WIDTH 16

// __device__ float __fadd_rd(float a, float b);
// __device__ float __fadd_ru(float a, float b);
// __device__ float __fmul_rd(float a, float b);
// __device__ float __fmul_ru(float a, float b);
// __device__ float __fdiv_rd(float a, float b);
// __device__ float fminf(float a, float b);
// __device__ float fmaxf(float a, float b);

namespace basic_ops
{
    template<typename T> inline __device__ T add_down  (T x, T y);
    template<typename T> inline __device__ T add_up    (T x, T y);
    template<typename T> inline __device__ T sub_down  (T x, T y);
    template<typename T> inline __device__ T sub_up    (T x, T y);
    template<typename T> inline __device__ T mul_down  (T x, T y);
    template<typename T> inline __device__ T mul_up    (T x, T y);
    template<typename T> inline __device__ T min       (T x, T y);
    template<typename T> inline __device__ T max       (T x, T y);

    template<> inline __device__ double add_down  (double x, double y) { return __dadd_rd(x, y); }
    template<> inline __device__ double add_up    (double x, double y) { return __dadd_ru(x, y); }
    template<> inline __device__ double mul_down  (double x, double y) { return __dmul_rd(x, y); }
    template<> inline __device__ double mul_up    (double x, double y) { return __dmul_ru(x, y); }
    template<> inline __device__ double min       (double x, double y) { return fmin(x, y); }
    template<> inline __device__ double max       (double x, double y) { return fmax(x, y); }

    template<> inline __device__ float add_down   (float x, float y)   { return __fadd_rd(x, y); } 
    template<> inline __device__ float add_up     (float x, float y)   { return __fadd_ru(x, y); }
    template<> inline __device__ float mul_down   (float x, float y)   { return __fmul_rd(x, y); }
    template<> inline __device__ float mul_up     (float x, float y)   { return __fmul_ru(x, y); }
    template<> inline __device__ float min        (float x, float y)   { return fminf(x, y); }
    template<> inline __device__ float max        (float x, float y)   { return fmaxf(x, y); }

}

namespace interval {
    template<typename T>
    struct __align__(8) Interval {
        T low;
        T high;
    };

    template<typename T> constexpr inline __device__ Interval<T> add(Interval<T> x, Interval<T> y) {
        Interval<T> z;
        z.low = basic_ops::add_down(x.low, y.low);
        z.high = basic_ops::add_up(x.high, y.high);
        return z;
    }

    template<typename T> constexpr inline __device__ Interval<T> mul(Interval<T> x, Interval<T> y) {
        Interval<T> z;
        z.low = basic_ops::min(basic_ops::min(basic_ops::mul_down(x.low, y.low), basic_ops::mul_down(x.low, y.high)), basic_ops::min(basic_ops::mul_down(x.high, y.low), basic_ops::mul_down(x.high, y.high)));
        z.high = basic_ops::max(basic_ops::max(basic_ops::mul_up(x.low, y.low), basic_ops::mul_up(x.low, y.high)), basic_ops::max(basic_ops::mul_up(x.high, y.low), basic_ops::mul_up(x.high, y.high)));
        return z;
    }
    template<typename T>
    struct IntervalMatrix {
        Interval<T>* data;
        int rows;
        int cols;
    };
    template<typename T>
    IntervalMatrix<T> InitDeviceMatrix(int rows, int cols) {
        IntervalMatrix<T> matrix;
        matrix.rows = rows;
        matrix.cols = cols;
        cudaMalloc(&matrix.data, rows * cols * sizeof(Interval<T>));
        return matrix;
    }

    template<typename T>
    void FreeDeviceMatrix(IntervalMatrix<T> matrix) {
        cudaFree(matrix.data);
    }

    template<typename T>
    IntervalMatrix<T> InitHostMatrix(int rows, int cols) {
        IntervalMatrix<T> matrix;
        matrix.rows = rows;
        matrix.cols = cols;
        matrix.data = new Interval<T>[rows * cols];
        return matrix;
    }

    template<typename T>
    void FreeHostMatrix(IntervalMatrix<T> matrix) {
        delete[] matrix.data;
    }
    template<typename T>
    void PrintIntervalMatrix(IntervalMatrix<T> A) {
        for (int i = 0; i < A.rows; i++) {
            for (int j = 0; j < A.cols; j++) {
                printf("[%f, %f] ", A.data[i * A.cols + j].low, A.data[i * A.cols + j].high);
            }
            printf("\n");
        }
    }
}

#endif // INTERVAL_HPP