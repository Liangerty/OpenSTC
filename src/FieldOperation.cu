#include "FieldOperation.cuh"

__device__ void cfd::compute_temperature(int i, int j, int k, const cfd::DParameter *param, cfd::DZone *zone) {
  const integer n_spec = param->n_spec;
  auto &Y = zone->sv;
  auto &Q = zone->cv;
  auto &bv = zone->bv;

  real mw{0};
  for (integer l = 0; l < n_spec; ++l) {
    mw += Y(i, j, k, l) / param->mw[l];
  }
  mw = 1 / mw;
  const real gas_const = R_u / mw;
  const real e = Q(i, j, k, 4) / Q(i, j, k, 0) - 0.5 *
                                                 (bv(i, j, k, 1) * bv(i, j, k, 1) + bv(i, j, k, 2) * bv(i, j, k, 2) +
                                                  bv(i, j, k, 3) * bv(i, j, k, 3));

  real err{1}, t{bv(i, j, k, 5)};
  constexpr integer max_iter{1000};
  constexpr real eps{1e-3};
  integer iter = 0;

  real h_i[MAX_SPEC_NUMBER], cp_i[MAX_SPEC_NUMBER];
  while (err > eps && iter++ < max_iter) {
    compute_enthalpy_and_cp(t, h_i, cp_i, param);
    real cp_tot{0}, h{0};
    for (integer l = 0; l < n_spec; ++l) {
      cp_tot += cp_i[l] * Y(i, j, k, l);
      h += h_i[l] * Y(i, j, k, l);
    }
    const real e_t = h - gas_const * t;
    const real cv = cp_tot - gas_const;
    const real t1 = t - (e_t - e) / cv;
    err = std::abs(1 - t1 / t);
    t = t1;
  }
  bv(i, j, k, 5) = t;
  bv(i, j, k, 4) = bv(i, j, k, 0) * t * gas_const;
}

__global__ void cfd::eliminate_k_gradient(cfd::DZone *zone) {
  const integer ngg{zone->ngg}, mx{zone->mx}, my{zone->my};
  integer i = (integer) (blockDim.x * blockIdx.x + threadIdx.x) - ngg;
  integer j = (integer) (blockDim.y * blockIdx.y + threadIdx.y) - ngg;
  if (i >= mx + ngg || j >= my + ngg) return;

  auto &bv = zone->bv;
  auto &sv = zone->sv;
  const integer n_scalar=zone->n_scal;

  for (integer k = 1; k <= ngg; ++k) {
    for (int l = 0; l < 6; ++l) {
      bv(i, j, k, l) = bv(i, j, 0, l);
      bv(i, j, -k, l) = bv(i, j, 0, l);
    }
    for (int l = 0; l < n_scalar; ++l) {
      sv(i, j, k, l) = sv(i, j, 0, l);
      sv(i, j, -k, l) = sv(i, j, 0, l);
    }
  }
}
