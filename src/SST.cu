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
  const real rhoK = zone->cv(i, j, k, n_spec + 5);
  const real tke = zone->sv(i, j, k, n_spec);
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
//  const real mut{zone->mut(i, j, k)};

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

__device__ void
compute_fv_2nd_order(DZone *zone, real *fv, DParameter *param, integer i, integer j, integer k, real xi_x, real xi_y,
                     real xi_z, real eta_x, real eta_y, real eta_z, real zeta_x, real zeta_y, real zeta_z, real mul,
                     real mut, real xi_x_div_jac, real xi_y_div_jac, real xi_z_div_jac) {
  const integer n_spec{param->n_spec};
  auto &sv = zone->sv;

  const real k_xi = sv(i + 1, j, k, n_spec) - sv(i, j, k, n_spec);
  const real k_eta =
      0.25 *
      (sv(i, j + 1, k, n_spec) - sv(i, j - 1, k, n_spec) + sv(i + 1, j + 1, k, n_spec) - sv(i + 1, j - 1, k, n_spec));
  const real k_zeta =
      0.25 *
      (sv(i, j, k + 1, n_spec) - sv(i, j, k - 1, n_spec) + sv(i + 1, j, k + 1, n_spec) - sv(i + 1, j, k - 1, n_spec));

  const real k_x = k_xi * xi_x + k_eta * eta_x + k_zeta * zeta_x;
  const real k_y = k_xi * xi_y + k_eta * eta_y + k_zeta * zeta_y;
  const real k_z = k_xi * xi_z + k_eta * eta_z + k_zeta * zeta_z;

  const real omega_xi = sv(i + 1, j, k, n_spec + 1) - sv(i, j, k, n_spec + 1);
  const real omega_eta =
      0.25 * (sv(i, j + 1, k, n_spec + 1) - sv(i, j - 1, k, n_spec + 1) + sv(i + 1, j + 1, k, n_spec + 1) -
              sv(i + 1, j - 1, k, n_spec + 1));
  const real omega_zeta = 0.25 *
                          (sv(i, j, k + 1, n_spec + 1) - sv(i, j, k - 1, n_spec + 1) + sv(i + 1, j, k + 1, n_spec + 1) -
                           sv(i + 1, j, k - 1, n_spec + 1));

  const real omega_x = omega_xi * xi_x + omega_eta * eta_x + omega_zeta * zeta_x;
  const real omega_y = omega_xi * xi_y + omega_eta * eta_y + omega_zeta * zeta_y;
  const real omega_z = omega_xi * xi_z + omega_eta * eta_z + omega_zeta * zeta_z;

  const real wall_dist = 0.5 * (zone->wall_distance(i, j, k) + zone->wall_distance(i + 1, j, k));

  const auto &pv = zone->bv;
  real f1{1};
  if (wall_dist > 1e-25) {
    const real km = 0.5 * (sv(i, j, k, n_spec) + sv(i + 1, j, k, n_spec));
    const real omegam = 0.5 * (sv(i, j, k, n_spec + 1) + sv(i + 1, j, k, n_spec + 1));
    const real param1{std::sqrt(km) / (0.09 * omegam * wall_dist)};

    const real rhom = 0.5 * (pv(i, j, k, 0) + pv(i + 1, j, k, 0));
    const real d2 = wall_dist * wall_dist;
    const real param2{500 * mul / (rhom * d2 * omegam)};
    const real CDkomega{max(1e-20, 2 * rhom * cfd::SST::sigma_omega2 / omegam *
                                   (k_x * omega_x + k_y * omega_y + k_z * omega_z))};
    const real param3{4 * rhom * SST::sigma_omega2 * km / (CDkomega * d2)};

    const real arg1{min(max(param1, param2), param3)};
    f1 = std::tanh(arg1 * arg1 * arg1 * arg1);
  }

  const real sigma_k = SST::sigma_k2 + SST::delta_sigma_k * f1;
  const real sigma_omega = SST::sigma_omega2 + SST::delta_sigma_omega * f1;

  fv[5 + n_spec] = (mul + mut * sigma_k) * (xi_x_div_jac * k_x + xi_y_div_jac * k_y + xi_z_div_jac * k_z);
  fv[6 + n_spec] =
      (mul + mut * sigma_omega) * (xi_x_div_jac * omega_x + xi_y_div_jac * omega_y + xi_z_div_jac * omega_z);
}

__device__ void
compute_gv_2nd_order(DZone *zone, real *gv, DParameter *param, integer i, integer j, integer k, real xi_x, real xi_y,
                     real xi_z, real eta_x, real eta_y, real eta_z, real zeta_x, real zeta_y, real zeta_z, real mul,
                     real mut, real eta_x_div_jac, real eta_y_div_jac, real eta_z_div_jac) {
  const integer n_spec{param->n_spec};
  auto &sv = zone->sv;

  const real k_xi =
      0.25 *
      (sv(i + 1, j, k, n_spec) - sv(i - 1, j, k, n_spec) + sv(i + 1, j + 1, k, n_spec) - sv(i - 1, j + 1, k, n_spec));
  const real k_eta = sv(i, j + 1, k, n_spec) - sv(i, j, k, n_spec);
  const real k_zeta =
      0.25 *
      (sv(i, j, k + 1, n_spec) - sv(i, j, k - 1, n_spec) + sv(i, j + 1, k + 1, n_spec) - sv(i, j + 1, k - 1, n_spec));

  const real k_x = k_xi * xi_x + k_eta * eta_x + k_zeta * zeta_x;
  const real k_y = k_xi * xi_y + k_eta * eta_y + k_zeta * zeta_y;
  const real k_z = k_xi * xi_z + k_eta * eta_z + k_zeta * zeta_z;

  const real omega_xi =
      0.25 * (sv(i + 1, j, k, n_spec + 1) - sv(i - 1, j, k, n_spec + 1) + sv(i + 1, j + 1, k, n_spec + 1) -
              sv(i - 1, j + 1, k, n_spec + 1));
  const real omega_eta = sv(i, j + 1, k, n_spec + 1) - sv(i, j, k, n_spec + 1);
  const real omega_zeta = 0.25 *
                          (sv(i, j, k + 1, n_spec + 1) - sv(i, j, k - 1, n_spec + 1) + sv(i, j + 1, k + 1, n_spec + 1) -
                           sv(i, j + 1, k - 1, n_spec + 1));

  const real omega_x = omega_xi * xi_x + omega_eta * eta_x + omega_zeta * zeta_x;
  const real omega_y = omega_xi * xi_y + omega_eta * eta_y + omega_zeta * zeta_y;
  const real omega_z = omega_xi * xi_z + omega_eta * eta_z + omega_zeta * zeta_z;

  const real wall_dist = 0.5 * (zone->wall_distance(i, j, k) + zone->wall_distance(i, j + 1, k));

  const auto &pv = zone->bv;
  real f1{1};
  if (wall_dist > 1e-25) {
    const real km = 0.5 * (sv(i, j, k, n_spec) + sv(i, j + 1, k, n_spec));
    const real omegam = 0.5 * (sv(i, j, k, n_spec + 1) + sv(i, j + 1, k, n_spec + 1));
    const real param1{std::sqrt(km) / (0.09 * omegam * wall_dist)};

    const real rhom = 0.5 * (pv(i, j, k, 0) + pv(i, j + 1, k, 0));
    const real d2 = wall_dist * wall_dist;
    const real param2{500 * mul / (rhom * d2 * omegam)};
    const real CDkomega{max(1e-20, 2 * rhom * SST::sigma_omega2 / omegam *
                                   (k_x * omega_x + k_y * omega_y + k_z * omega_z))};
    const real param3{4 * rhom * SST::sigma_omega2 * km / (CDkomega * d2)};

    const real arg1{min(max(param1, param2), param3)};
    f1 = std::tanh(arg1 * arg1 * arg1 * arg1);
  }

  const real sigma_k = SST::sigma_k2 + SST::delta_sigma_k * f1;
  const real sigma_omega = SST::sigma_omega2 + SST::delta_sigma_omega * f1;

  gv[5 + n_spec] = (mul + mut * sigma_k) * (eta_x_div_jac * k_x + eta_y_div_jac * k_y + eta_z_div_jac * k_z);
  gv[6 + n_spec] =
      (mul + mut * sigma_omega) * (eta_x_div_jac * omega_x + eta_y_div_jac * omega_y + eta_z_div_jac * omega_z);
}

__device__ void
compute_hv_2nd_order(DZone *zone, real *hv, DParameter *param, integer i, integer j, integer k, real xi_x, real xi_y,
                     real xi_z, real eta_x, real eta_y, real eta_z, real zeta_x, real zeta_y, real zeta_z, real mul,
                     real mut, real zeta_x_div_jac, real zeta_y_div_jac, real zeta_z_div_jac) {
  const integer n_spec{param->n_spec};
  auto &sv = zone->sv;

  const real k_xi =
      0.25 *
      (sv(i + 1, j, k, n_spec) - sv(i - 1, j, k, n_spec) + sv(i + 1, j, k + 1, n_spec) - sv(i - 1, j, k + 1, n_spec));
  const real k_eta =
      0.25 *
      (sv(i, j + 1, k, n_spec) - sv(i, j - 1, k, n_spec) + sv(i, j + 1, k + 1, n_spec) - sv(i, j - 1, k + 1, n_spec));
  const real k_zeta = sv(i, j, k + 1, n_spec) - sv(i, j, k, n_spec);

  const real k_x = k_xi * xi_x + k_eta * eta_x + k_zeta * zeta_x;
  const real k_y = k_xi * xi_y + k_eta * eta_y + k_zeta * zeta_y;
  const real k_z = k_xi * xi_z + k_eta * eta_z + k_zeta * zeta_z;

  const real omega_xi =
      0.25 * (sv(i + 1, j, k, n_spec + 1) - sv(i - 1, j, k, n_spec + 1) + sv(i + 1, j, k + 1, n_spec + 1) -
              sv(i - 1, j, k + 1, n_spec + 1));
  const real omega_eta =
      0.25 * (sv(i, j + 1, k, n_spec + 1) - sv(i, j - 1, k, n_spec + 1) + sv(i, j + 1, k + 1, n_spec + 1) -
              sv(i, j - 1, k + 1, n_spec + 1));
  const real omega_zeta = sv(i, j, k + 1, n_spec + 1) - sv(i, j, k, n_spec + 1);

  const real omega_x = omega_xi * xi_x + omega_eta * eta_x + omega_zeta * zeta_x;
  const real omega_y = omega_xi * xi_y + omega_eta * eta_y + omega_zeta * zeta_y;
  const real omega_z = omega_xi * xi_z + omega_eta * eta_z + omega_zeta * zeta_z;

  const real wall_dist = 0.5 * (zone->wall_distance(i, j, k) + zone->wall_distance(i, j, k + 1));

  const auto &pv = zone->bv;
  real f1{1};
  if (wall_dist > 1e-25) {
    const real km = 0.5 * (sv(i, j, k, n_spec) + sv(i, j, k + 1, n_spec));
    const real omegam = 0.5 * (sv(i, j, k, n_spec + 1) + sv(i, j, k + 1, n_spec + 1));
    const real param1{std::sqrt(km) / (0.09 * omegam * wall_dist)};

    const real rhom = 0.5 * (pv(i, j, k, 0) + pv(i, j, k + 1, 0));
    const real d2 = wall_dist * wall_dist;
    const real param2{500 * mul / (rhom * d2 * omegam)};
    const real CDkomega{max(1e-20, 2 * rhom * SST::sigma_omega2 / omegam *
                                   (k_x * omega_x + k_y * omega_y + k_z * omega_z))};
    const real param3{4 * rhom * SST::sigma_omega2 * km / (CDkomega * d2)};

    const real arg1{min(max(param1, param2), param3)};
    f1 = std::tanh(arg1 * arg1 * arg1 * arg1);
  }

  const real sigma_k = SST::sigma_k2 + SST::delta_sigma_k * f1;
  const real sigma_omega = SST::sigma_omega2 + SST::delta_sigma_omega * f1;

  hv[5 + n_spec] = (mul + mut * sigma_k) * (zeta_x_div_jac * k_x + zeta_y_div_jac * k_y + zeta_z_div_jac * k_z);
  hv[6 + n_spec] =
      (mul + mut * sigma_omega) *
      (zeta_x_div_jac * omega_x + zeta_y_div_jac * omega_y + zeta_z_div_jac * omega_z);
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