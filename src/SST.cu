#include "SST.cuh"
#include "Field.h"
#include "Transport.cuh"

namespace cfd::SST {
__device__ void compute_mut(cfd::DZone *zone, integer i, integer j, integer k, real mul) {
  const auto &m = zone->metric(i, j, k);
  const real xi_x{m(1, 1)}, xi_y{m(1, 2)}, xi_z{m(1, 3)};
  const real eta_x{m(2, 1)}, eta_y{m(2, 2)}, eta_z{m(2, 3)};
  const real zeta_x{m(3, 1)}, zeta_y{m(3, 2)}, zeta_z{m(3, 3)};

  // Compute the gradient of velocity
  const auto &bv = zone->bv;
  const real u_y = 0.5 * (xi_y * (bv(i + 1, j, k, 1) - bv(i - 1, j, k, 1)) +
                          eta_y * (bv(i, j + 1, k, 1) - bv(i, j - 1, k, 1)) +
                          zeta_y * (bv(i, j, k + 1, 1) - bv(i, j, k - 1, 1)));
  const real u_z = 0.5 * (xi_z * (bv(i + 1, j, k, 1) - bv(i - 1, j, k, 1)) +
                          eta_z * (bv(i, j + 1, k, 1) - bv(i, j - 1, k, 1)) +
                          zeta_z * (bv(i, j, k + 1, 1) - bv(i, j, k - 1, 1)));
  const real v_x = 0.5 * (xi_x * (bv(i + 1, j, k, 2) - bv(i - 1, j, k, 2)) +
                          eta_x * (bv(i, j + 1, k, 2) - bv(i, j - 1, k, 2)) +
                          zeta_x * (bv(i, j, k + 1, 2) - bv(i, j, k - 1, 2)));
  const real v_z = 0.5 * (xi_z * (bv(i + 1, j, k, 2) - bv(i - 1, j, k, 2)) +
                          eta_z * (bv(i, j + 1, k, 2) - bv(i, j - 1, k, 2)) +
                          zeta_z * (bv(i, j, k + 1, 2) - bv(i, j, k - 1, 2)));
  const real w_x = 0.5 * (xi_x * (bv(i + 1, j, k, 3) - bv(i - 1, j, k, 3)) +
                          eta_x * (bv(i, j + 1, k, 3) - bv(i, j - 1, k, 3)) +
                          zeta_x * (bv(i, j, k + 1, 3) - bv(i, j, k - 1, 3)));
  const real w_y = 0.5 * (xi_y * (bv(i + 1, j, k, 3) - bv(i - 1, j, k, 3)) +
                          eta_y * (bv(i, j + 1, k, 3) - bv(i, j - 1, k, 3)) +
                          zeta_y * (bv(i, j, k + 1, 3) - bv(i, j, k - 1, 3)));

  // First, compute the turbulent viscosity.
  // Theoretically, this should be computed after updating the basic variables, but after that we won't need it until now.
  // Besides, we need the velocity gradients in the computation, which are also needed when computing source terms.
  // In order to alleviate the computational burden, we put the computation of mut here.
  const integer n_spec{zone->n_spec};
  const real rhoK = zone->cv(i, j, k, n_spec + 5);
  const real tke = zone->sv(i, j, k, n_spec);
  const real omega = zone->sv(i, j, k, n_spec + 1);
  const real vorticity = std::sqrt((v_x - u_y) * (v_x - u_y) + (w_x - u_z) * (w_x - u_z) + (w_y - v_z) * (w_y - v_z));
  const real density = zone->bv(i, j, k, 0);

  // If wall, mut=0. Else, compute mut as in the if statement.
  real f2{1};
  const real dy = zone->wall_distance(i, j, k);
  if (dy > 1e-25) {
    const real param1 = 2 * std::sqrt(tke) / (0.09 * omega * dy);
    const real param2 = 500 * mul / (density * dy * dy * omega);
    const real arg2 = max(param1, param2);
    f2 = std::tanh(arg2 * arg2);
  }
  real mut{0};
  if (const real denominator = max(SST::a_1 * omega, vorticity * f2); denominator > 1e-25) {
    mut = SST::a_1 * rhoK / denominator;
  }
  zone->mut(i, j, k) = mut;
}

__device__ void compute_source_and_mut(cfd::DZone *zone, integer i, integer j, integer k, DParameter *param) {
  const integer n_spec{zone->n_spec};

  const auto &m = zone->metric(i, j, k);
  const real xi_x{m(1, 1)}, xi_y{m(1, 2)}, xi_z{m(1, 3)};
  const real eta_x{m(2, 1)}, eta_y{m(2, 2)}, eta_z{m(2, 3)};
  const real zeta_x{m(3, 1)}, zeta_y{m(3, 2)}, zeta_z{m(3, 3)};

  // Compute the gradient of velocity
  const auto &bv = zone->bv;
  const real u_x = 0.5 * (xi_x * (bv(i + 1, j, k, 1) - bv(i - 1, j, k, 1)) +
                          eta_x * (bv(i, j + 1, k, 1) - bv(i, j - 1, k, 1)) +
                          zeta_x * (bv(i, j, k + 1, 1) - bv(i, j, k - 1, 1)));
  const real u_y = 0.5 * (xi_y * (bv(i + 1, j, k, 1) - bv(i - 1, j, k, 1)) +
                          eta_y * (bv(i, j + 1, k, 1) - bv(i, j - 1, k, 1)) +
                          zeta_y * (bv(i, j, k + 1, 1) - bv(i, j, k - 1, 1)));
  const real u_z = 0.5 * (xi_z * (bv(i + 1, j, k, 1) - bv(i - 1, j, k, 1)) +
                          eta_z * (bv(i, j + 1, k, 1) - bv(i, j - 1, k, 1)) +
                          zeta_z * (bv(i, j, k + 1, 1) - bv(i, j, k - 1, 1)));
  const real v_x = 0.5 * (xi_x * (bv(i + 1, j, k, 2) - bv(i - 1, j, k, 2)) +
                          eta_x * (bv(i, j + 1, k, 2) - bv(i, j - 1, k, 2)) +
                          zeta_x * (bv(i, j, k + 1, 2) - bv(i, j, k - 1, 2)));
  const real v_y = 0.5 * (xi_y * (bv(i + 1, j, k, 2) - bv(i - 1, j, k, 2)) +
                          eta_y * (bv(i, j + 1, k, 2) - bv(i, j - 1, k, 2)) +
                          zeta_y * (bv(i, j, k + 1, 2) - bv(i, j, k - 1, 2)));
  const real v_z = 0.5 * (xi_z * (bv(i + 1, j, k, 2) - bv(i - 1, j, k, 2)) +
                          eta_z * (bv(i, j + 1, k, 2) - bv(i, j - 1, k, 2)) +
                          zeta_z * (bv(i, j, k + 1, 2) - bv(i, j, k - 1, 2)));
  const real w_x = 0.5 * (xi_x * (bv(i + 1, j, k, 3) - bv(i - 1, j, k, 3)) +
                          eta_x * (bv(i, j + 1, k, 3) - bv(i, j - 1, k, 3)) +
                          zeta_x * (bv(i, j, k + 1, 3) - bv(i, j, k - 1, 3)));
  const real w_y = 0.5 * (xi_y * (bv(i + 1, j, k, 3) - bv(i - 1, j, k, 3)) +
                          eta_y * (bv(i, j + 1, k, 3) - bv(i, j - 1, k, 3)) +
                          zeta_y * (bv(i, j, k + 1, 3) - bv(i, j, k - 1, 3)));
  const real w_z = 0.5 * (xi_z * (bv(i + 1, j, k, 3) - bv(i - 1, j, k, 3)) +
                          eta_z * (bv(i, j + 1, k, 3) - bv(i, j - 1, k, 3)) +
                          zeta_z * (bv(i, j, k + 1, 3) - bv(i, j, k - 1, 3)));
  const real density = zone->bv(i, j, k, 0);
  const real omega = zone->sv(i, j, k, n_spec + 1);
  auto &sv = zone->sv;
  const real k_x = 0.5 * (xi_x * (sv(i + 1, j, k, n_spec) - sv(i - 1, j, k, n_spec)) +
                          eta_x * (sv(i, j + 1, k, n_spec) - sv(i, j - 1, k, n_spec)) +
                          zeta_x * (sv(i, j, k + 1, n_spec) - sv(i, j, k - 1, n_spec)));
  const real k_y = 0.5 * (xi_y * (sv(i + 1, j, k, n_spec) - sv(i - 1, j, k, n_spec)) +
                          eta_y * (sv(i, j + 1, k, n_spec) - sv(i, j - 1, k, n_spec)) +
                          zeta_y * (sv(i, j, k + 1, n_spec) - sv(i, j, k - 1, n_spec)));
  const real k_z = 0.5 * (xi_z * (sv(i + 1, j, k, n_spec) - sv(i - 1, j, k, n_spec)) +
                          eta_z * (sv(i, j + 1, k, n_spec) - sv(i, j - 1, k, n_spec)) +
                          zeta_z * (sv(i, j, k + 1, n_spec) - sv(i, j, k - 1, n_spec)));

  const real omega_x = 0.5 * (xi_x * (sv(i + 1, j, k, n_spec + 1) - sv(i - 1, j, k, n_spec + 1)) +
                              eta_x * (sv(i, j + 1, k, n_spec + 1) - sv(i, j - 1, k, n_spec + 1)) +
                              zeta_x * (sv(i, j, k + 1, n_spec + 1) - sv(i, j, k - 1, n_spec + 1)));
  const real omega_y = 0.5 * (xi_y * (sv(i + 1, j, k, n_spec + 1) - sv(i - 1, j, k, n_spec + 1)) +
                              eta_y * (sv(i, j + 1, k, n_spec + 1) - sv(i, j - 1, k, n_spec + 1)) +
                              zeta_y * (sv(i, j, k + 1, n_spec + 1) - sv(i, j, k - 1, n_spec + 1)));
  const real omega_z = 0.5 * (xi_z * (sv(i + 1, j, k, n_spec + 1) - sv(i - 1, j, k, n_spec + 1)) +
                              eta_z * (sv(i, j + 1, k, n_spec + 1) - sv(i, j - 1, k, n_spec + 1)) +
                              zeta_z * (sv(i, j, k + 1, n_spec + 1) - sv(i, j, k - 1, n_spec + 1)));
  const real inter_var =
      2 * density * cfd::SST::sigma_omega2 / omega * (k_x * omega_x + k_y * omega_y + k_z * omega_z);

  // First, compute the turbulent viscosity.
  // Theoretically, this should be computed after updating the basic variables, but after that we won't need it until now.
  // Besides, we need the velocity gradients in the computation, which are also needed when computing source terms.
  // In order to alleviate the computational burden, we put the computation of mut here.
  const real tke = zone->sv(i, j, k, n_spec);
  const real rhoK = density * tke;
  const real vorticity = std::sqrt((v_x - u_y) * (v_x - u_y) + (w_x - u_z) * (w_x - u_z) + (w_y - v_z) * (w_y - v_z));

  // If wall, mut=0. Else, compute mut as in the if statement.
  real f1{1}, f2{1};
  const real dy = zone->wall_distance(i, j, k);
  if (dy > 1e-25) {
    const real param1{std::sqrt(tke) / (0.09 * omega * dy)};

    const real d2 = dy * dy;
    const real param2{500 * zone->mul(i, j, k) / (density * d2 * omega)};
    const real arg2 = max(2 * param1, param2);
    f2 = std::tanh(arg2 * arg2);

    const real CDkomega{max(1e-20, inter_var)};
    const real param3{4 * density * SST::sigma_omega2 * tke / (CDkomega * d2)};

    const real arg1{min(max(param1, param2), param3)};
    f1 = std::tanh(arg1 * arg1 * arg1 * arg1);
  }
  real mut{0};
  if (const real denominator = max(SST::a_1 * omega, vorticity * f2); denominator > 1e-25) {
    mut = SST::a_1 * rhoK / denominator;
  }
  zone->mut(i, j, k) = mut;

  const real beta = SST::beta_2 + SST::delta_beta * f1;
  if (mut > 1e-25) {
    // Next, compute the source term for turbulent kinetic energy.
    const real divU = u_x + v_y + w_z;
    const real divU2 = divU * divU;

    const real prod_k =
        mut * (2 * (u_x * u_x + v_y * v_y + w_z * w_z) - 2.0 / 3 * divU2 + (u_y + v_x) * (u_y + v_x) +
               (u_z + w_x) * (u_z + w_x) + (v_z + w_y) * (v_z + w_y)) - 2.0 / 3 * rhoK * divU;
    const real diss_k = SST::beta_star * rhoK * omega;
    const real jac = zone->jac(i, j, k);
    auto &dq = zone->dq;
    dq(i, j, k, n_spec + 5) += jac * (prod_k - diss_k);

    // omega source term
    const real gamma = SST::gamma2 + SST::delta_gamma * f1;
    const real prod_omega = gamma * density / mut * prod_k + (1 - f1) * inter_var;
    const real diss_omega = beta * density * omega * omega;
    dq(i, j, k, n_spec + 6) += jac * (prod_omega - diss_omega);
  }

  if (param->turb_implicit == 1) {
    zone->turb_src_jac(i, j, k, 0) = -2 * SST::beta_star * omega;
    zone->turb_src_jac(i, j, k, 1) = -2 * beta * omega;
  }
}

__global__ void implicit_treat(cfd::DZone *zone) {
  const integer extent[3]{zone->mx, zone->my, zone->mz};
  const integer i = blockDim.x * blockIdx.x + threadIdx.x;
  const integer j = blockDim.y * blockIdx.y + threadIdx.y;
  const integer k = blockDim.z * blockIdx.z + threadIdx.z;
  if (i >= extent[0] || j >= extent[1] || k >= extent[2]) return;

  // Used in explicit temporal scheme
  const integer n_spec{zone->n_spec};
  auto &dq = zone->dq;
  const real dt_local = zone->dt_local(i, j, k);
  const auto &src_jac = zone->turb_src_jac;
  dq(i, j, k, n_spec + 5) /= 1 - dt_local * src_jac(i, j, k, 0);
  dq(i, j, k, n_spec + 6) /= 1 - dt_local * src_jac(i, j, k, 1);
}

__device__ void implicit_treat_for_dq0(cfd::DZone *zone, real diag, integer i, integer j, integer k) {
  // Used in DPLUR, called from device
  const integer n_spec{zone->n_spec};
  auto &dq = zone->dq;
  const real dt_local = zone->dt_local(i, j, k);
  const auto &src_jac = zone->turb_src_jac;
  dq(i, j, k, n_spec + 5) /= diag - dt_local * src_jac(i, j, k, 0);
  dq(i, j, k, n_spec + 6) /= diag - dt_local * src_jac(i, j, k, 1);
}

__device__ void
implicit_treat_for_dqk(cfd::DZone *zone, real diag, integer i, integer j, integer k, const real *dq_total) {
  // Used in DPLUR, called from device
  const integer n_spec{zone->n_spec};
  auto &dqk = zone->dqk;
  const auto &dq0 = zone->dq0;
  const real dt_local = zone->dt_local(i, j, k);
  const auto &src_jac = zone->turb_src_jac;
  dqk(i, j, k, n_spec + 5) =
      dq0(i, j, k, n_spec + 5) + dt_local * dq_total[5 + n_spec] / (diag - dt_local * src_jac(i, j, k, 0));
  dqk(i, j, k, n_spec + 6) =
      dq0(i, j, k, n_spec + 6) + dt_local * dq_total[6 + n_spec] / (diag - dt_local * src_jac(i, j, k, 1));
}

}