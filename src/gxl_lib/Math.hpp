#pragma once
// #include "Matrix.hpp"
// #include <span>
// #include <vector>
// #include <numeric>
#include <cmath>

namespace gxl{
template <typename T>
#ifdef __CUDACC__
__host__ __device__
#endif
int sgn(T a) {
  return a < 0 ? -1 : 1;
}

template <typename T>
int del(T a, T b) {
  return std::abs(a) == std::abs(b) ? 1 : 0;
}

// template <typename T>
// void solve_linear_eqn(MatrixDyn<T>& a, std::span<T> b) {
//   const int dim{a.n_col()};
//   std::vector ipiv(dim, 0);
//   std::iota(ipiv.begin(), ipiv.end(), 0);

//   // Column pivot LU decomposition
//   for (int k = 0; k < dim; ++k) {
//     int ik{k};
//     for (int i = k; i < dim; ++i) {
//       for (int t = 0; t < k; ++t)
//         a(i, k) -= a(i, t) * a(t, k);
//       if (std::abs(a(i, k)) > std::abs(a(ik, k)))
//         ik = i;
//     }
//     ipiv[k] = ik;
//     if (ik != k) {
//       for (int t = 0; t < dim; ++t)
//         std::swap(a(k, t), a(ik, t));
//     }
//     for (int j = k + 1; j < dim; ++j) {
//       for (int t = 0; t < k; ++t)
//         a(k, j) -= a(k, t) * a(t, j);
//     }
//     for (int i = k + 1; i < dim; ++i)
//       a(i, k) /= a(k, k);
//   }

//   // Solve the linear system with LU matrix
//   for (int k = 0; k < dim; ++k) {
//     int t = ipiv[k];
//     if (t != k)
//       std::swap(b[k], b[t]);
//   }
//   for (int i = 1; i < dim; ++i) {
//     for (int t = 0; t < i; ++t)
//       b[i] -= a(i, t) * b[t];
//   }
//   b[dim - 1] /= a(dim - 1, dim - 1);
//   for (int i = dim - 2; i >= 0; --i) {
//     for (int t = i + 1; t < dim; ++t)
//       b[i] -= a(i, t) * b[t];
//     b[i] /= a(i, i);
//   }
// }
}
