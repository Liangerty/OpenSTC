#pragma once

#include "DParameter.h"

namespace cfd {
struct DZone;

__device__ void finite_rate_chemistry(DZone *zone, integer i, integer j, integer k, DParameter *param);

__device__ void forward_reaction_rate(real t, real *kf, const real *concentration, DParameter *param);

__device__ real arrhenius(real t, real A, real b, real Ea);

__device__ void
backward_reaction_rate(real t, const real *kf, const real *concentration, const DParameter *param, real *kb);

__device__ void
rate_of_progress(const real *kf, const real *kb, const real *c, real *q, real *q1, real *q2, const DParameter *param);

__device__ void chemical_source(const real *q1, const real *q2, real *omega_d, real *omega, const DParameter *param);

__device__ void
compute_chem_src_jacobian(DZone *zone, integer i, integer j, integer k, const DParameter *param, const real *q1,
                          const real *q2);

__device__ void
compute_chem_src_jacobian_diagonal(DZone *zone, integer i, integer j, integer k, const DParameter *param,
                                   const real *q1);

__global__ void EPI(DZone *zone);

__device__ void EPI_for_dq0(DZone *zone, real diag, integer i, integer j, integer k);
__device__ void EPI_for_dqk(DZone *zone, real diag, integer i, integer j, integer k, const real* dq_total);

__device__ void solve_chem_system(real *lhs, DZone *zone, integer i, integer j, integer k);
__device__ void solve_chem_system(real *lhs, real *rhs, integer dim);

__global__ void DA(DZone *zone);

} // cfd
