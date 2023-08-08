#include "ViscousScheme.cuh"
#include "Field.h"
#include "Thermo.cuh"
#include "DParameter.h"

namespace cfd {
__device__ SecOrdViscScheme::SecOrdViscScheme() : ViscousScheme() {

}

__device__ void SecOrdViscScheme::compute_fv(integer idx[3], DZone *zone, real *fv, DParameter *param) {
  const auto i = idx[0], j = idx[1], k = idx[2];
  const auto &m = zone->metric(i, j, k);
  const auto &m1 = zone->metric(i + 1, j, k);

  const real xi_x = 0.5 * (m(1, 1) + m1(1, 1));
  const real xi_y = 0.5 * (m(1, 2) + m1(1, 2));
  const real xi_z = 0.5 * (m(1, 3) + m1(1, 3));
  const real eta_x = 0.5 * (m(2, 1) + m1(2, 1));
  const real eta_y = 0.5 * (m(2, 2) + m1(2, 2));
  const real eta_z = 0.5 * (m(2, 3) + m1(2, 3));
  const real zeta_x = 0.5 * (m(3, 1) + m1(3, 1));
  const real zeta_y = 0.5 * (m(3, 2) + m1(3, 2));
  const real zeta_z = 0.5 * (m(3, 3) + m1(3, 3));

  // 1st order partial derivative of velocity to computational coordinate
  const auto& pv=zone->bv;
  const real u_xi  = pv(i + 1, j, k, 1) - pv(i, j, k, 1);
  const real u_eta = 0.25 * (pv(i, j + 1, k, 1) - pv(i, j - 1, k, 1) + pv(i + 1, j + 1, k, 1) - pv(i + 1, j - 1, k, 1));
  const real u_zeta = 0.25 * (pv(i, j, k + 1, 1) - pv(i, j, k - 1, 1) + pv(i + 1, j, k + 1, 1) - pv(i + 1, j, k - 1, 1));
  const real v_xi  = pv(i + 1, j, k, 2) - pv(i, j, k, 2);
  const real v_eta = 0.25 * (pv(i, j + 1, k, 2) - pv(i, j - 1, k, 2) + pv(i + 1, j + 1, k, 2) - pv(i + 1, j - 1, k, 2));
  const real v_zeta = 0.25 * (pv(i, j, k + 1, 2) - pv(i, j, k - 1, 2) + pv(i + 1, j, k + 1, 2) - pv(i + 1, j, k - 1, 2));
  const real w_xi  = pv(i + 1, j, k, 3) - pv(i, j, k, 3);
  const real w_eta = 0.25 * (pv(i, j + 1, k, 3) - pv(i, j - 1, k, 3) + pv(i + 1, j + 1, k, 3) - pv(i + 1, j - 1, k, 3));
  const real w_zeta = 0.25 * (pv(i, j, k + 1, 3) - pv(i, j, k - 1, 3) + pv(i + 1, j, k + 1, 3) - pv(i + 1, j, k - 1, 3));
  const real t_xi  = pv(i + 1, j, k, 5) - pv(i, j, k, 5);
  const real t_eta = 0.25 * (pv(i, j + 1, k, 5) - pv(i, j - 1, k, 5) + pv(i + 1, j + 1, k, 5) - pv(i + 1, j - 1, k, 5));
  const real t_zeta = 0.25 * (pv(i, j, k + 1, 5) - pv(i, j, k - 1, 5) + pv(i + 1, j, k + 1, 5) - pv(i + 1, j, k - 1, 5));

  // chain rule for derivative
  const real u_x = u_xi * xi_x + u_eta * eta_x + u_zeta * zeta_x;
  const real u_y = u_xi * xi_y + u_eta * eta_y + u_zeta * zeta_y;
  const real u_z = u_xi * xi_z + u_eta * eta_z + u_zeta * zeta_z;
  const real v_x = v_xi * xi_x + v_eta * eta_x + v_zeta * zeta_x;
  const real v_y = v_xi * xi_y + v_eta * eta_y + v_zeta * zeta_y;
  const real v_z = v_xi * xi_z + v_eta * eta_z + v_zeta * zeta_z;
  const real w_x = w_xi * xi_x + w_eta * eta_x + w_zeta * zeta_x;
  const real w_y = w_xi * xi_y + w_eta * eta_y + w_zeta * zeta_y;
  const real w_z = w_xi * xi_z + w_eta * eta_z + w_zeta * zeta_z;

  const real viscosity=0.5*(zone->mul(i,j,k)+zone->mul(i+1,j,k));

  // Compute the viscous stress
  const real tau_xx = viscosity * (4 * u_x - 2 * v_y - 2 * w_z) / 3.0;
  const real tau_yy = viscosity * (4 * v_y - 2 * u_x - 2 * w_z) / 3.0;
  const real tau_zz = viscosity * (4 * w_z - 2 * u_x - 2 * v_y) / 3.0;
  const real tau_xy = viscosity * (u_y + v_x);
  const real tau_xz = viscosity * (u_z + w_x);
  const real tau_yz = viscosity * (v_z + w_y);

  const real xi_x_div_jac = 0.5 * (m(1, 1) * zone->jac(i, j, k) + m1(1, 1) * zone->jac(i + 1, j, k));
  const real xi_y_div_jac = 0.5 * (m(1, 2) * zone->jac(i, j, k) + m1(1, 2) * zone->jac(i + 1, j, k));
  const real xi_z_div_jac = 0.5 * (m(1, 3) * zone->jac(i, j, k) + m1(1, 3) * zone->jac(i + 1, j, k));

  fv[0]=0;
  fv[1] = xi_x_div_jac * tau_xx + xi_y_div_jac * tau_xy + xi_z_div_jac * tau_xz;
  fv[2] = xi_x_div_jac * tau_xy + xi_y_div_jac * tau_yy + xi_z_div_jac * tau_yz;
  fv[3] = xi_x_div_jac * tau_xz + xi_y_div_jac * tau_yz + xi_z_div_jac * tau_zz;

  const real um           = 0.5 * (pv(i, j, k, 1) + pv(i + 1, j, k, 1));
  const real vm           = 0.5 * (pv(i, j, k, 2) + pv(i + 1, j, k, 2));
  const real wm           = 0.5 * (pv(i, j, k, 3) + pv(i + 1, j, k, 3));
  const real t_x          = t_xi * xi_x + t_eta * eta_x + t_zeta * zeta_x;
  const real t_y          = t_xi * xi_y + t_eta * eta_y + t_zeta * zeta_y;
  const real t_z          = t_xi * xi_z + t_eta * eta_z + t_zeta * zeta_z;
  const real conductivity = 0.5 * (zone->conductivity(i, j, k) + zone->conductivity(i + 1, j, k));

  fv[4] = um * fv[1] + vm * fv[2] + wm * fv[3] + conductivity * (xi_x_div_jac * t_x + xi_y_div_jac * t_y + xi_z_div_jac * t_z);

#if MULTISPECIES==1
  const integer n_spec{zone->n_spec};
  const auto& y=zone->yk;

  real h[MAX_SPEC_NUMBER], diffusivity[MAX_SPEC_NUMBER], y_x[MAX_SPEC_NUMBER], y_y[MAX_SPEC_NUMBER], y_z[MAX_SPEC_NUMBER];
  const real tm = 0.5 * (pv(i, j, k, 5) + pv(i + 1, j, k, 5));
  compute_enthalpy(tm, h, param);
  real rho_uc{0}, rho_vc{0}, rho_wc{0};
  for (int l = 0; l < n_spec; ++l) {
    diffusivity[l] = 0.5 * (zone->rho_D(i, j, k, l) + zone->rho_D(i + 1, j, k, l));

    const real y_xi  = y(i + 1, j, k, l) - y(i, j, k, l);
    const real y_eta = 0.25 * (y(i, j + 1, k, l) - y(i, j - 1, k, l) + y(i + 1, j + 1, k, l) - y(i + 1, j - 1, k, l));
    const real y_zeta = 0.25 * (y(i, j, k + 1, l) - y(i, j, k - 1, l) + y(i + 1, j, k + 1, l) - y(i + 1, j, k - 1, l));

    y_x[l] = y_xi * xi_x + y_eta * eta_x + y_zeta * zeta_x;
    y_y[l] = y_xi * xi_y + y_eta * eta_y + y_zeta * zeta_y;
    y_z[l] = y_xi * xi_z + y_eta * eta_z + y_zeta * zeta_z;

    // Add the influence of species diffusion
    const auto DMulH = diffusivity[l] * h[l];
    fv[4] += xi_x_div_jac * DMulH * y_x[l];
    fv[4] += xi_y_div_jac * DMulH * y_y[l];
    fv[4] += xi_z_div_jac * DMulH * y_z[l];

    rho_uc += diffusivity[l] * y_x[l];
    rho_vc += diffusivity[l] * y_y[l];
    rho_wc += diffusivity[l] * y_z[l];
  }
  for (int l = 0; l < n_spec; ++l) {
    fv[5 + l] = diffusivity[l] * (xi_x_div_jac * y_x[l] + xi_y_div_jac * y_y[l] + xi_z_div_jac * y_z[l]) - 0.5 * (y(i, j, k, l) + y(i + 1, j, k, l)) * (xi_x_div_jac * rho_uc + xi_y_div_jac * rho_vc + xi_z_div_jac * rho_wc);
  }
#endif
}

__device__ void SecOrdViscScheme::compute_gv(integer *idx, DZone *zone, real *gv, cfd::DParameter *param) {
  const auto i = idx[0], j = idx[1], k = idx[2];
  const auto &m = zone->metric(i, j, k);
  const auto &m1 = zone->metric(i, j + 1, k);

  const real xi_x = 0.5 * (m(1, 1) + m1(1, 1));
  const real xi_y = 0.5 * (m(1, 2) + m1(1, 2));
  const real xi_z = 0.5 * (m(1, 3) + m1(1, 3));
  const real eta_x = 0.5 * (m(2, 1) + m1(2, 1));
  const real eta_y = 0.5 * (m(2, 2) + m1(2, 2));
  const real eta_z = 0.5 * (m(2, 3) + m1(2, 3));
  const real zeta_x = 0.5 * (m(3, 1) + m1(3, 1));
  const real zeta_y = 0.5 * (m(3, 2) + m1(3, 2));
  const real zeta_z = 0.5 * (m(3, 3) + m1(3, 3));

  // 1st order partial derivative of velocity to computational coordinate
  const auto& pv=zone->bv;
  const real u_xi = 0.25 * (pv(i + 1, j, k, 1) - pv(i - 1, j, k, 1) + pv(i + 1, j + 1, k, 1) - pv(i - 1, j + 1, k, 1));
  const real u_eta  = pv(i, j + 1, k, 1) - pv(i, j, k, 1);
  const real u_zeta = 0.25 * (pv(i, j, k + 1, 1) - pv(i, j, k - 1, 1) + pv(i, j + 1, k + 1, 1) - pv(i, j + 1, k - 1, 1));
  const real v_xi = 0.25 * (pv(i + 1, j, k, 2) - pv(i - 1, j, k, 2) + pv(i + 1, j + 1, k, 2) - pv(i - 1, j + 1, k, 2));
  const real v_eta  = pv(i, j + 1, k, 2) - pv(i, j, k, 2);
  const real v_zeta = 0.25 * (pv(i, j, k + 1, 2) - pv(i, j, k - 1, 2) + pv(i, j + 1, k + 1, 2) - pv(i, j + 1, k - 1, 2));
  const real w_xi = 0.25 * (pv(i + 1, j, k, 3) - pv(i - 1, j, k, 3) + pv(i + 1, j + 1, k, 3) - pv(i - 1, j + 1, k, 3));
  const real w_eta  = pv(i, j + 1, k, 3) - pv(i, j, k, 3);
  const real w_zeta = 0.25 * (pv(i, j, k + 1, 3) - pv(i, j, k - 1, 3) + pv(i, j + 1, k + 1, 3) - pv(i, j + 1, k - 1, 3));
  const real t_xi = 0.25 * (pv(i + 1, j, k, 5) - pv(i - 1, j, k, 5) + pv(i + 1, j + 1, k, 5) - pv(i - 1, j + 1, k, 5));
  const real t_eta  = pv(i, j + 1, k, 5) - pv(i, j, k, 5);
  const real t_zeta = 0.25 * (pv(i, j, k + 1, 5) - pv(i, j, k - 1, 5) + pv(i, j + 1, k + 1, 5) - pv(i, j + 1, k - 1, 5));

  // chain rule for derivative
  const real u_x = u_xi * xi_x + u_eta * eta_x + u_zeta * zeta_x;
  const real u_y = u_xi * xi_y + u_eta * eta_y + u_zeta * zeta_y;
  const real u_z = u_xi * xi_z + u_eta * eta_z + u_zeta * zeta_z;
  const real v_x = v_xi * xi_x + v_eta * eta_x + v_zeta * zeta_x;
  const real v_y = v_xi * xi_y + v_eta * eta_y + v_zeta * zeta_y;
  const real v_z = v_xi * xi_z + v_eta * eta_z + v_zeta * zeta_z;
  const real w_x = w_xi * xi_x + w_eta * eta_x + w_zeta * zeta_x;
  const real w_y = w_xi * xi_y + w_eta * eta_y + w_zeta * zeta_y;
  const real w_z = w_xi * xi_z + w_eta * eta_z + w_zeta * zeta_z;

  const real viscosity=0.5*(zone->mul(i,j,k)+zone->mul(i,j+1,k));

  // Compute the viscous stress
  const real tau_xx = viscosity * (4 * u_x - 2 * v_y - 2 * w_z) / 3.0;
  const real tau_yy = viscosity * (4 * v_y - 2 * u_x - 2 * w_z) / 3.0;
  const real tau_zz = viscosity * (4 * w_z - 2 * u_x - 2 * v_y) / 3.0;
  const real tau_xy = viscosity * (u_y + v_x);
  const real tau_xz = viscosity * (u_z + w_x);
  const real tau_yz = viscosity * (v_z + w_y);

  const real eta_x_div_jac = 0.5 * (m(2, 1) * zone->jac(i, j, k) + m1(2, 1) * zone->jac(i, j + 1, k));
  const real eta_y_div_jac = 0.5 * (m(2, 2) * zone->jac(i, j, k) + m1(2, 2) * zone->jac(i, j + 1, k));
  const real eta_z_div_jac = 0.5 * (m(2, 3) * zone->jac(i, j, k) + m1(2, 3) * zone->jac(i, j + 1, k));

  gv[0]=0;
  gv[1] = eta_x_div_jac * tau_xx + eta_y_div_jac * tau_xy + eta_z_div_jac * tau_xz;
  gv[2] = eta_x_div_jac * tau_xy + eta_y_div_jac * tau_yy + eta_z_div_jac * tau_yz;
  gv[3] = eta_x_div_jac * tau_xz + eta_y_div_jac * tau_yz + eta_z_div_jac * tau_zz;

  const real um           = 0.5 * (pv(i, j, k, 1) + pv(i, j + 1, k, 1));
  const real vm           = 0.5 * (pv(i, j, k, 2) + pv(i, j + 1, k, 2));
  const real wm           = 0.5 * (pv(i, j, k, 3) + pv(i, j + 1, k, 3));
  const real t_x          = t_xi * xi_x + t_eta * eta_x + t_zeta * zeta_x;
  const real t_y          = t_xi * xi_y + t_eta * eta_y + t_zeta * zeta_y;
  const real t_z          = t_xi * xi_z + t_eta * eta_z + t_zeta * zeta_z;
  const real conductivity = 0.5 * (zone->conductivity(i, j, k) + zone->conductivity(i, j + 1, k));

  gv[4] = um * gv[1] + vm * gv[2] + wm * gv[3] + conductivity * (eta_x_div_jac * t_x + eta_y_div_jac * t_y + eta_z_div_jac * t_z);

#if MULTISPECIES==1
  const integer n_spec{zone->n_spec};
  const auto& y=zone->yk;

  real h[MAX_SPEC_NUMBER], diffusivity[MAX_SPEC_NUMBER], y_x[MAX_SPEC_NUMBER], y_y[MAX_SPEC_NUMBER], y_z[MAX_SPEC_NUMBER];
  const real tm = 0.5 * (pv(i, j, k, 5) + pv(i, j + 1, k, 5));
  compute_enthalpy(tm, h, param);
  real rho_uc{0}, rho_vc{0}, rho_wc{0};
  for (int l = 0; l < n_spec; ++l) {
    diffusivity[l] = 0.5 * (zone->rho_D(i, j, k, l) + zone->rho_D(i, j + 1, k, l));

    const real y_xi = 0.25 * (y(i + 1, j, k, l) - y(i - 1, j, k, l) + y(i + 1, j + 1, k, l) - y(i - 1, j + 1, k, l));
    const real y_eta  = y(i, j + 1, k, l) - y(i, j, k, l);
    const real y_zeta = 0.25 * (y(i, j, k + 1, l) - y(i, j, k - 1, l) + y(i, j + 1, k + 1, l) - y(i, j + 1, k - 1, l));

    y_x[l] = y_xi * xi_x + y_eta * eta_x + y_zeta * zeta_x;
    y_y[l] = y_xi * xi_y + y_eta * eta_y + y_zeta * zeta_y;
    y_z[l] = y_xi * xi_z + y_eta * eta_z + y_zeta * zeta_z;

    // Add the influence of species diffusion
    const auto DMulH = diffusivity[l] * h[l];
    gv[4] += eta_x_div_jac * DMulH * y_x[l];
    gv[4] += eta_y_div_jac * DMulH * y_y[l];
    gv[4] += eta_z_div_jac * DMulH * y_z[l];

    rho_uc += diffusivity[l] * y_x[l];
    rho_vc += diffusivity[l] * y_y[l];
    rho_wc += diffusivity[l] * y_z[l];
  }
  for (int l = 0; l < n_spec; ++l) {
    gv[5 + l] = diffusivity[l] * (eta_x_div_jac * y_x[l] + eta_y_div_jac * y_y[l] + eta_z_div_jac * y_z[l]) - 0.5 * (y(i, j, k, l) + y(i, j + 1, k, l)) * (eta_x_div_jac * rho_uc + eta_y_div_jac * rho_vc + eta_z_div_jac * rho_wc);
  }
#endif
}

__device__ void SecOrdViscScheme::compute_hv(integer *idx, DZone *zone, real *hv, cfd::DParameter *param) {
  const auto i = idx[0], j = idx[1], k = idx[2];
  const auto &m = zone->metric(i, j, k);
  const auto &m1 = zone->metric(i, j, k + 1);

  const real xi_x = 0.5 * (m(1, 1) + m1(1, 1));
  const real xi_y = 0.5 * (m(1, 2) + m1(1, 2));
  const real xi_z = 0.5 * (m(1, 3) + m1(1, 3));
  const real eta_x = 0.5 * (m(2, 1) + m1(2, 1));
  const real eta_y = 0.5 * (m(2, 2) + m1(2, 2));
  const real eta_z = 0.5 * (m(2, 3) + m1(2, 3));
  const real zeta_x = 0.5 * (m(3, 1) + m1(3, 1));
  const real zeta_y = 0.5 * (m(3, 2) + m1(3, 2));
  const real zeta_z = 0.5 * (m(3, 3) + m1(3, 3));

  // 1st order partial derivative of velocity to computational coordinate
  const auto& pv=zone->bv;
  const real u_xi = 0.25 * (pv(i + 1, j, k, 1) - pv(i - 1, j, k, 1) + pv(i + 1, j, k + 1, 1) - pv(i - 1, j, k + 1, 1));
  const real u_eta = 0.25 * (pv(i, j + 1, k, 1) - pv(i, j - 1, k, 1) + pv(i, j + 1, k + 1, 1) - pv(i, j - 1, k + 1, 1));
  const real u_zeta = pv(i, j, k + 1, 1) - pv(i, j, k, 1);
  const real v_xi   = 0.25 * (pv(i + 1, j, k, 2) - pv(i - 1, j, k, 2) + pv(i + 1, j, k + 1, 2) - pv(i - 1, j, k + 1, 2));
  const real v_eta = 0.25 * (pv(i, j + 1, k, 2) - pv(i, j - 1, k, 2) + pv(i, j + 1, k + 1, 2) - pv(i, j - 1, k + 1, 2));
  const real v_zeta = pv(i, j, k + 1, 2) - pv(i, j, k, 2);
  const real w_xi   = 0.25 * (pv(i + 1, j, k, 3) - pv(i - 1, j, k, 3) + pv(i + 1, j, k + 1, 3) - pv(i - 1, j, k + 1, 3));
  const real w_eta = 0.25 * (pv(i, j + 1, k, 3) - pv(i, j - 1, k, 3) + pv(i, j + 1, k + 1, 3) - pv(i, j - 1, k + 1, 3));
  const real w_zeta = pv(i, j, k + 1, 3) - pv(i, j, k, 3);
  const real t_xi   = 0.25 * (pv(i + 1, j, k, 5) - pv(i - 1, j, k, 5) + pv(i + 1, j, k + 1, 5) - pv(i - 1, j, k + 1, 5));
  const real t_eta = 0.25 * (pv(i, j + 1, k, 5) - pv(i, j - 1, k, 5) + pv(i, j + 1, k + 1, 5) - pv(i, j - 1, k + 1, 5));
  const real t_zeta = pv(i, j, k + 1, 5) - pv(i, j, k, 5);

  // chain rule for derivative
  const real u_x = u_xi * xi_x + u_eta * eta_x + u_zeta * zeta_x;
  const real u_y = u_xi * xi_y + u_eta * eta_y + u_zeta * zeta_y;
  const real u_z = u_xi * xi_z + u_eta * eta_z + u_zeta * zeta_z;
  const real v_x = v_xi * xi_x + v_eta * eta_x + v_zeta * zeta_x;
  const real v_y = v_xi * xi_y + v_eta * eta_y + v_zeta * zeta_y;
  const real v_z = v_xi * xi_z + v_eta * eta_z + v_zeta * zeta_z;
  const real w_x = w_xi * xi_x + w_eta * eta_x + w_zeta * zeta_x;
  const real w_y = w_xi * xi_y + w_eta * eta_y + w_zeta * zeta_y;
  const real w_z = w_xi * xi_z + w_eta * eta_z + w_zeta * zeta_z;

  const real viscosity=0.5*(zone->mul(i,j,k)+zone->mul(i,j,k+1));

  // Compute the viscous stress
  const real tau_xx = viscosity * (4 * u_x - 2 * v_y - 2 * w_z) / 3.0;
  const real tau_yy = viscosity * (4 * v_y - 2 * u_x - 2 * w_z) / 3.0;
  const real tau_zz = viscosity * (4 * w_z - 2 * u_x - 2 * v_y) / 3.0;
  const real tau_xy = viscosity * (u_y + v_x);
  const real tau_xz = viscosity * (u_z + w_x);
  const real tau_yz = viscosity * (v_z + w_y);

  const real zeta_x_div_jac = 0.5 * (m(3, 1) * zone->jac(i, j, k) + m1(3, 1) * zone->jac(i, j, k + 1));
  const real zeta_y_div_jac = 0.5 * (m(3, 2) * zone->jac(i, j, k) + m1(3, 2) * zone->jac(i, j, k + 1));
  const real zeta_z_div_jac = 0.5 * (m(3, 3) * zone->jac(i, j, k) + m1(3, 3) * zone->jac(i, j, k + 1));

  hv[0]=0;
  hv[1] = zeta_x_div_jac * tau_xx + zeta_y_div_jac * tau_xy + zeta_z_div_jac * tau_xz;
  hv[2] = zeta_x_div_jac * tau_xy + zeta_y_div_jac * tau_yy + zeta_z_div_jac * tau_yz;
  hv[3] = zeta_x_div_jac * tau_xz + zeta_y_div_jac * tau_yz + zeta_z_div_jac * tau_zz;

  const real um           = 0.5 * (pv(i, j, k, 1) + pv(i, j, k + 1, 1));
  const real vm           = 0.5 * (pv(i, j, k, 2) + pv(i, j, k + 1, 2));
  const real wm           = 0.5 * (pv(i, j, k, 3) + pv(i, j, k + 1, 3));
  const real t_x          = t_xi * xi_x + t_eta * eta_x + t_zeta * zeta_x;
  const real t_y          = t_xi * xi_y + t_eta * eta_y + t_zeta * zeta_y;
  const real t_z          = t_xi * xi_z + t_eta * eta_z + t_zeta * zeta_z;
  const real conductivity = 0.5 * (zone->conductivity(i, j, k) + zone->conductivity(i, j, k + 1));

  hv[4] = um * hv[1] + vm * hv[2] + wm * hv[3] + conductivity * (zeta_x_div_jac * t_x + zeta_y_div_jac * t_y + zeta_z_div_jac * t_z);

#if MULTISPECIES==1
  const integer n_spec{zone->n_spec};
  const auto& y=zone->yk;

  real h[MAX_SPEC_NUMBER], diffusivity[MAX_SPEC_NUMBER], y_x[MAX_SPEC_NUMBER], y_y[MAX_SPEC_NUMBER], y_z[MAX_SPEC_NUMBER];
  const real tm = 0.5 * (pv(i, j, k, 5) + pv(i, j, k + 1, 5));
  compute_enthalpy(tm, h, param);
  real rho_uc{0}, rho_vc{0}, rho_wc{0};
  for (int l = 0; l < n_spec; ++l) {
    diffusivity[l] = 0.5 * (zone->rho_D(i, j, k, l) + zone->rho_D(i, j + 1, k, l));

    const real y_xi = 0.25 * (y(i + 1, j, k, l) - y(i - 1, j, k, l) + y(i + 1, j, k + 1, l) - y(i - 1, j, k + 1, l));
    const real y_eta = 0.25 * (y(i, j + 1, k, l) - y(i, j - 1, k, l) + y(i, j + 1, k + 1, l) - y(i, j - 1, k + 1, l));
    const real y_zeta = y(i, j, k + 1, l) - y(i, j, k, l);

    y_x[l] = y_xi * xi_x + y_eta * eta_x + y_zeta * zeta_x;
    y_y[l] = y_xi * xi_y + y_eta * eta_y + y_zeta * zeta_y;
    y_z[l] = y_xi * xi_z + y_eta * eta_z + y_zeta * zeta_z;

    // Add the influence of species diffusion
    const auto DMulH = diffusivity[l] * h[l];
    hv[4] += zeta_x_div_jac * DMulH * y_x[l];
    hv[4] += zeta_y_div_jac * DMulH * y_y[l];
    hv[4] += zeta_z_div_jac * DMulH * y_z[l];

    rho_uc += diffusivity[l] * y_x[l];
    rho_vc += diffusivity[l] * y_y[l];
    rho_wc += diffusivity[l] * y_z[l];
  }
  for (int l = 0; l < n_spec; ++l) {
    hv[5 + l] = diffusivity[l] * (zeta_x_div_jac * y_x[l] + zeta_y_div_jac * y_y[l] + zeta_z_div_jac * y_z[l]) - 0.5 * (y(i, j, k, l) + y(i, j, k + 1, l)) * (zeta_x_div_jac * rho_uc + zeta_y_div_jac * rho_vc + zeta_z_div_jac * rho_wc);
  }
#endif
}

__device__ void ViscousScheme::compute_fv(integer *idx, DZone *zone, real *fv, DParameter *param) {}

__device__ void ViscousScheme::compute_gv(integer *idx, DZone *zone, real *gv, DParameter *param) {}

__device__ void ViscousScheme::compute_hv(integer *idx, DZone *zone, real *hv, DParameter *param) {}
} // cfd