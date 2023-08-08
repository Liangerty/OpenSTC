#pragma once
#ifdef __CUDACC__

#include <cuda_runtime.h>
#include <cstdio>

#endif

#include <vector>

enum class Major {
  ColMajor = 0,
  RowMajor
};

#ifdef __CUDACC__
namespace ggxl {
template<typename T, Major major = Major::ColMajor>
class Array3D {
private:
  int disp1 = 0, disp2 = 0, dispt = 0;
  int sz = 0;
  T *val = nullptr;
  int ng = 0;
  int n1 = 0, n2 = 0, n3 = 0;

public:

  cudaError_t allocate_memory(int dim1, int dim2, int dim3, int n_ghost = 0);

  __device__ T &operator()(const int i, const int j, const int k) {
    if constexpr (major == Major::ColMajor) {
      return val[k * disp1 + j * disp2 + i + dispt];
    } else {
      return val[i * disp1 + j * disp2 + k + dispt];
    }
  }

  __device__ const T &operator()(const int i, const int j, const int k) const {
    if constexpr (major == Major::ColMajor) {
      return val[k * disp1 + j * disp2 + i + dispt];
    } else {
      return val[i * disp1 + j * disp2 + k + dispt];
    }
  }

  T *data() { return val; }

  auto size() { return sz; }
};

template<typename T, Major major>
inline cudaError_t Array3D<T, major>::allocate_memory(int dim1, int dim2, int dim3, int n_ghost) {
  ng = n_ghost;
  n1 = dim1;
  n2 = dim2;
  n3 = dim3;
  if constexpr (major == Major::ColMajor) {
    disp2 = n1 + 2 * ng;
  } else {
    disp2 = n3 + 2 * ng;
  }
  disp1 = (n2 + 2 * ng) * disp2;
  dispt = (disp1 + disp2 + 1) * ng;
  sz = (n1 + 2 * ng) * (n2 + 2 * ng) * (n3 + 2 * ng);
  cudaError_t err = cudaMalloc(&val, sz * sizeof(T));
  return err;
}

template<typename T, Major major = Major::ColMajor>
class VectorField3D {
private:
  int disp1 = 0, disp2 = 0, dispt = 0;
  int sz = 0;
  T *val = nullptr;
  int ng = 0;
  int n1 = 0, n2 = 0, n3 = 0, n4 = 0;

public:

  cudaError_t allocate_memory(int dim1, int dim2, int dim3, int dim4, int n_ghost = 0);

  __device__ T &operator()(const int i, const int j, const int k, const int l) {
    if constexpr (major == Major::RowMajor) {
      return val[i * disp1 + j * disp2 + k * n4 + dispt + l];
    } else {
      return val[k * disp1 + j * disp2 + i + dispt + l * sz];
    }
  }

  __device__ const T &operator()(const int i, const int j, const int k, const int l) const {
    if constexpr (major == Major::RowMajor) {
      return val[i * disp1 + j * disp2 + k * n4 + dispt + l];
    } else {
      return val[k * disp1 + j * disp2 + i + dispt + l * sz];
    }
  }

  T *operator[](int l) {
    static_assert(major == Major::ColMajor);
    return val + l * sz;
//    return &val[l * sz];
  }

  T *data() { return val; }

  auto size() { return sz; }
};

template<typename T, Major major>
inline cudaError_t
VectorField3D<T, major>::allocate_memory(int dim1, int dim2, int dim3, int dim4, int n_ghost) {
  ng = n_ghost;
  n1 = dim1;
  n2 = dim2;
  n3 = dim3;
  n4 = dim4;
  if constexpr (major == Major::RowMajor) {
    disp2 = (n3 + 2 * ng) * n4;
    disp1 = (n2 + 2 * ng) * disp2;
    dispt = (disp1 + disp2 + n4) * ng;
  } else { // Column major
    disp2 = n1 + 2 * ng;
    disp1 = (n2 + 2 * ng) * disp2;
    dispt = (disp1 + disp2 + 1) * ng;
  }
  sz = (n1 + 2 * ng) * (n2 + 2 * ng) * (n3 + 2 * ng);
  cudaError_t err = cudaMalloc(&val, sz * n4 * sizeof(T));
  return err;
}

template<typename T, Major major = Major::ColMajor>
class VectorField3DHost {
  int disp2{0}, disp1{0}, dispt{0};
  T *data_ = nullptr;
  int ng{0}, n1{0}, n2{0}, n3{0}, n4{0}, sz{0};
//  std::vector<T> data_;
public:
//  explicit VectorField3DHost(int dim1, int dim2, int dim3, int dim4, int n_ghost);

  auto data() const { return data_; }
//  auto data() const { return data_.data(); }

  auto data() { return data_; }
//  auto data() { return data_.data(); }

  /**
   * \brief Get the l-th variable at position (i,j,k)
   * \param i x index
   * \param j y index
   * \param k z index
   * \param l variable index in a vector
   * \return the l-th variable at position (i,j,k)
   */
  T &operator()(const int i, const int j, const int k, const int l) {
    if constexpr (major == Major::RowMajor) {
      return data_[i * disp1 + j * disp2 + k * n4 + dispt + l];
    } else {
      return data_[k * disp1 + j * disp2 + i + dispt + l * sz];
    }
  }

  T *operator[](int l) {
    static_assert(major == Major::ColMajor);
    return &data_[l * sz];
  }

  void resize(int ni, int nj, int nk, int nl, int ngg);

  int n_var() const { return n4; }
};

template<typename T, Major major>
void VectorField3DHost<T, major>::resize(int ni, int nj, int nk, int nl, int ngg) {
  ng = ngg;
  n1 = ni + 2 * ngg;
  n2 = nj + 2 * ngg;
  n3 = nk + 2 * ngg;
  n4 = nl;
  sz = n1 * n2 * n3;
  if constexpr (major == Major::RowMajor) {
    disp2 = n3 * n4;
    disp1 = n2 * disp2;
    dispt = (disp1 + disp2 + n4) * ng;
  } else {
    disp2 = n1;
    disp1 = n2 * disp2;
    dispt = (disp1 + disp2 + 1) * ng;
  }
  cudaError_t err = cudaHostAlloc(&data_, sz * n4 * sizeof(T), cudaHostAllocDefault);
  if (err != cudaSuccess) {
    printf(
        "The VectorField3DHost isn't allocated by cudaHostAlloc, not enough page-locked memory. Use malloc instead\n");
    data_ = (real *) malloc(sz * n4 * sizeof(T));
  }
//  data_.resize(n1 * n2 * n3 * n4, t);
  n1 = ni;
  n2 = nj;
  n3 = nk;
}

}
#endif

namespace gxl {
/**
 * \brief a 3D array containing ghost elements in all directions.
 * \details for a given size mx, my, mz and a given ghost value ng, the array contains (mx+2ng)*(my+2ng)*(mz+2ng) elements
 *  and the index are allowed to be negative for the ghost elements
 * \tparam T type of containing elements
 */
template<typename T, Major major = Major::ColMajor>
class Array3D {
  int ng{0}, n1{0}, n2{0}, n3{0};
  int disp2{0}, disp1{0}, dispt{0};
  std::vector<T> data_;
public:
  explicit Array3D(const int ni = 0, const int nj = 0, const int nk = 0, const int _n_ghost = 0, T dd = T{}) :
      ng(_n_ghost), n1(ni), n2(nj), n3(nk), disp2(n1 + 2 * ng), disp1((n2 + 2 * ng) * disp2),
      dispt((disp1 + disp2 + 1) * ng),
      data_((ni + 2 * ng) * (nj + 2 * ng) * (nk + 2 * ng), dd) {
    if constexpr (major == Major::RowMajor) {
      disp2 = n3 + 2 * ng;
      disp1 = (n2 + 2 * ng) * disp2;
      dispt = (disp1 + disp2 + 1) * ng;
    }
  }

  Array3D(const Array3D &arr);

  T &operator()(const int i, const int j, const int k) {
    if constexpr (major == Major::RowMajor) {
      return data_[i * disp1 + j * disp2 + k + dispt];
    } else {
      return data_[k * disp1 + j * disp2 + i + dispt];
    }
  }

  const T &operator()(const int i, const int j, const int k) const {
    if constexpr (major == Major::RowMajor) {
      return data_[i * disp1 + j * disp2 + k + dispt];
    } else {
      return data_[k * disp1 + j * disp2 + i + dispt];
    }
  }

  T *data() { return data_.data(); }

  const T *data() const { return data_.data(); }

  auto begin() { return data_.begin(); }

  auto end() { return data_.end(); }

  void resize(int ni, int nj, int nk, int _n_ghost, T dd = T{});
};

template<typename T, Major major>
inline Array3D<T, major>::Array3D(const Array3D &arr): Array3D<T, major>(arr.n1, arr.n2, arr.n3, arr.ng) {
  data_ = arr.data_;
}

template<typename DataType, Major major>
void Array3D<DataType, major>::resize(const int ni, const int nj, const int nk, const int _n_ghost, DataType dd) {
  ng = _n_ghost;
  n1 = ni;
  n2 = nj;
  n3 = nk;
  data_.resize((ni + 2 * ng) * (nj + 2 * ng) * (nk + 2 * ng), dd);
  if constexpr (major == Major::RowMajor) {
    disp2 = n3 + 2 * ng;
  } else {
    disp2 = n1 + 2 * ng;
  }
  disp1 = (n2 + 2 * ng) * disp2;
  dispt = (disp1 + disp2 + 1) * ng;
}

/**
 * \brief As its name, should have 4 indexes. First 3 index supply the spatial position and the 4th index is used for
 *  vector subscript. Thus, the ghost grid is only assigned for first 3 index.
 * \tparam T the data type of the stored datas
 */
template<typename T, Major major = Major::ColMajor>
class VectorField3D {
  int ng{0}, n1{0}, n2{0}, n3{0}, n4{0}, sz{0};
  std::vector<T> data_;
  int disp2{0}, disp1{0}, dispt{0};
public:
  auto data() const { return data_.data(); }

  auto data() { return data_.data(); }

  /**
   * \brief Get the l-th variable at position (i,j,k)
   * \param i x index
   * \param j y index
   * \param k z index
   * \param l variable index in a vector
   * \return the l-th variable at position (i,j,k)
   */
  T &operator()(const int i, const int j, const int k, const int l) {
    if constexpr (major == Major::RowMajor) {
      return data_[i * disp1 + j * disp2 + k * n4 + dispt + l];
    } else {
      return data_[k * disp1 + j * disp2 + i + dispt + l * sz];
    }
  }

  T *operator[](int l) {
    static_assert(major == Major::ColMajor);
    return &data_[l * sz];
  }

  void resize(int ni, int nj, int nk, int nl, int ngg, T &&t = T{});

  int n_var() const { return n4; }
};

template<typename T, Major major>
void VectorField3D<T, major>::resize(int ni, int nj, int nk, int nl, int ngg, T &&t) {
  ng = ngg;
  n1 = ni + 2 * ngg;
  n2 = nj + 2 * ngg;
  n3 = nk + 2 * ngg;
  n4 = nl;
  sz = n1 * n2 * n3;
  if constexpr (major == Major::RowMajor) {
    disp2 = n3 * n4;
    disp1 = n2 * disp2;
    dispt = (disp1 + disp2 + n4) * ng;
  } else {
    disp2 = n1;
    disp1 = n2 * disp2;
    dispt = (disp1 + disp2 + 1) * ng;
  }
  data_.resize(n1 * n2 * n3 * n4, t);
  n1 = ni;
  n2 = nj;
  n3 = nk;
}

}
