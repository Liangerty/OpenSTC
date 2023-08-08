#pragma once
#include "Define.h"

namespace cfd {
struct DParameter;

__device__ void first_order_reconstruct(const real *pv, real *pv_l, real *pv_r, integer idx_shared, integer n_var);

__device__ void
MUSCL_reconstruct(const real *pv, real *pv_l, real *pv_r, integer idx_shared, integer n_var, integer limiter);

__device__ void NND2_reconstruct(const real *pv, real *pv_l, real *pv_r, integer idx_shared, integer n_var, integer limiter);

} // cfd
