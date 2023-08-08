#pragma once

#include "Define.h"
#include "DParameter.h"

namespace cfd {
struct DZone;

namespace SST {
// Model constants
static constexpr double beta_star = 0.09;
static constexpr double sqrt_beta_star = 0.3;
static constexpr double kappa = 0.41;
// SST inner parameters, the first group:
static constexpr double sigma_k1 = 0.85;
static constexpr double sigma_omega1 = 0.5;
static constexpr double beta_1 = 0.0750;
static constexpr double a_1 = 0.31;
static constexpr double gamma1 = beta_1 / beta_star - sigma_omega1 * kappa * kappa / sqrt_beta_star;

// k-epsilon parameters, the second group:
static constexpr double sigma_k2 = 1;
static constexpr double sigma_omega2 = 0.856;
static constexpr double beta_2 = 0.0828;
static constexpr double gamma2 = beta_2 / beta_star - sigma_omega2 * kappa * kappa / sqrt_beta_star;

// Mixed parameters, their difference, used in computations
static constexpr double delta_sigma_k = sigma_k1 - sigma_k2;
static constexpr double delta_sigma_omega = sigma_omega1 - sigma_omega2;
static constexpr double delta_beta = beta_1 - beta_2;
static constexpr double delta_gamma = gamma1 - gamma2;

__device__ void compute_mut(cfd::DZone *zone, integer i, integer j, integer k, real mul);

__device__ void compute_source_and_mut(cfd::DZone *zone, integer i, integer j, integer k, DParameter *param);

// Warning: There too many arguments passes here, which may degrade the performance compared to previous inline version.
// However, this makes the code more compact and easier for later models to be added on.
// We may investigate how we can accelerate this without sacrificing the compactness of the code later.
// E.g., maybe some variables can be put on shared memory? Which kind of memory can be reached with the called device kernel?
// Because of the calling overhead, the following codes are inlined manually, which caused a 4% performance upgrade.
__device__ void
compute_fv_2nd_order(DZone *zone, real *fv, DParameter *param, integer i, integer j, integer k, real xi_x, real xi_y,
                     real xi_z, real eta_x, real eta_y, real eta_z, real zeta_x, real zeta_y, real zeta_z, real mul,
                     real mut, real xi_x_div_jac, real xi_y_div_jac, real xi_z_div_jac);

__device__ void
compute_gv_2nd_order(DZone *zone, real *gv, DParameter *param, integer i, integer j, integer k, real xi_x, real xi_y,
                     real xi_z, real eta_x, real eta_y, real eta_z, real zeta_x, real zeta_y, real zeta_z, real mul,
                     real mut, real eta_x_div_jac, real eta_y_div_jac, real eta_z_div_jac);

__device__ void
compute_hv_2nd_order(DZone *zone, real *hv, DParameter *param, integer i, integer j, integer k, real xi_x, real xi_y,
                     real xi_z, real eta_x, real eta_y, real eta_z, real zeta_x, real zeta_y, real zeta_z, real mul,
                     real mut, real zeta_x_div_jac, real zeta_y_div_jac, real zeta_z_div_jac);

__global__ void implicit_treat(DZone *zone);

__device__ void implicit_treat_for_dq0(DZone *zone, real diag, integer i, integer j, integer k);

__device__ void implicit_treat_for_dqk(DZone *zone, real diag, integer i, integer j, integer k, const real *dq_total);
}
}