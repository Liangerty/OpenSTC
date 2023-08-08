#include "Transport.cuh"
#include "DParameter.h"
#include "Field.h"
#include "Constants.h"

__host__ __device__
real cfd::Sutherland(real temperature) {
  return 1.716e-5 * pow(temperature / 273, 1.5) * (273 + 111) / (temperature + 111);
}

real cfd::compute_viscosity(real temperature, real mw_total, real const *Y, Species &spec) {
  // This method can only be used on CPU, while for GPU the allocation may be performed in every step
  for (int i = 0; i < spec.n_spec; ++i) {
    spec.x[i] = Y[i] * mw_total / spec.mw[i];
    const real t_dl{temperature * spec.LJ_potent_inv[i]};  // dimensionless temperature
    const real collision_integral{1.147 * std::pow(t_dl, -0.145) + std::pow(t_dl + 0.5, -2)};
    spec.vis_spec[i] = spec.vis_coeff[i] * std::sqrt(temperature) / collision_integral;
  }
  for (int i = 0; i < spec.n_spec; ++i) {
    for (int j = 0; j < spec.n_spec; ++j) {
      if (i == j) {
        spec.partition_fun(i, j) = 1.0;
      } else {
        const real numerator{1 + std::sqrt(spec.vis_spec[i] / spec.vis_spec[j]) * spec.WjDivWi_to_One4th(i, j)};
        spec.partition_fun(i, j) = numerator * numerator * spec.sqrt_WiDivWjPl1Mul8(i, j);
      }
    }
  }
  real viscosity{0};
  for (int i = 0; i < spec.n_spec; ++i) {
    real vis_temp{0};
    for (int j = 0; j < spec.n_spec; ++j) {
      vis_temp += spec.partition_fun(i, j) * spec.x[j];
    }
    viscosity += spec.vis_spec[i] * spec.x[i] / vis_temp;
  }
  return viscosity;
}

__device__ void
cfd::compute_transport_property(integer i, integer j, integer k, real temperature, real mw_total, const real *cp,
                                cfd::DParameter *param, DZone *zone) {
  const auto n_spec{param->n_spec};
  const real *mw = param->mw;
  const auto yk = zone->sv;

  real X[MAX_SPEC_NUMBER], vis[MAX_SPEC_NUMBER];
  for (int l = 0; l < n_spec; ++l) {
    X[l] = yk(i, j, k, l) * mw_total / mw[l];
    const real t_dl{temperature * param->LJ_potent_inv[l]}; //dimensionless temperature
    const real collision_integral{1.147 * std::pow(t_dl, -0.145) + std::pow(t_dl + 0.5, -2)};
    vis[l] = param->vis_coeff[l] * std::sqrt(temperature) / collision_integral;
  }

  real viscosity = 0;
  real conductivity = 0;
  for (int m = 0; m < n_spec; ++m) {
    real vis_temp{0};
    for (int n = 0; n < n_spec; ++n) {
      real partition_func{1.0};
      if (m != n) {
        const real numerator{1 + std::sqrt(vis[m] / vis[n]) * param->WjDivWi_to_One4th(m, n)};
        partition_func = numerator * numerator * param->sqrt_WiDivWjPl1Mul8(m, n);
      }
      vis_temp += partition_func * X[n];
    }
    const real cond_temp = 1.065 * vis_temp - 0.065 * X[m];
    viscosity += vis[m] * X[m] / vis_temp;
    const real lambda = vis[m] * (cp[m] + 1.25 * R_u / mw[m]);
    conductivity += lambda * X[m] / cond_temp;
  }
  zone->mul(i, j, k) = viscosity;
  zone->thermal_conductivity(i, j, k) = conductivity;

  // The diffusivity is now computed via constant Schmidt number method
  const real sc{param->Sc};
  for (auto l = 0; l < n_spec; ++l) {
    if (std::abs(X[l] - 1) < 1e-3) {
      zone->rho_D(i, j, k, l) = viscosity / sc;
    } else {
      zone->rho_D(i, j, k, l) = (1 - yk(i, j, k, l)) * viscosity / ((1 - X[l]) * sc);
    }
  }
}

__device__ real
cfd::compute_viscosity(integer i, integer j, integer k, real temperature, real mw_total, cfd::DParameter *param,
                       DZone *zone) {
  const auto n_spec{param->n_spec};
  const real *mw = param->mw;
  const auto &yk = zone->sv;

  real X[MAX_SPEC_NUMBER], vis[MAX_SPEC_NUMBER];
  for (int l = 0; l < n_spec; ++l) {
    X[l] = yk(i, j, k, l) * mw_total / mw[l];
    const real t_dl{temperature * param->LJ_potent_inv[l]}; //dimensionless temperature
    const real collision_integral{1.147 * std::pow(t_dl, -0.145) + std::pow(t_dl + 0.5, -2)};
    vis[l] = param->vis_coeff[l] * std::sqrt(temperature) / collision_integral;
  }

  real viscosity = 0;
  for (int m = 0; m < n_spec; ++m) {
    real vis_temp{0};
    for (int n = 0; n < n_spec; ++n) {
      real partition_func{1.0};
      if (m != n) {
        const real numerator{1 + std::sqrt(vis[m] / vis[n]) * param->WjDivWi_to_One4th(m, n)};
        partition_func = numerator * numerator * param->sqrt_WiDivWjPl1Mul8(m, n);
      }
      vis_temp += partition_func * X[n];
    }
    viscosity += vis[m] * X[m] / vis_temp;
  }
  return viscosity;
}
