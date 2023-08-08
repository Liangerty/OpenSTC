#pragma once

#include "Define.h"
#include "Reconstruction.cuh"
#include "Constants.h"
#include "DParameter.h"
#include "Thermo.cuh"

namespace cfd {
template<MixtureModel mix_model, TurbMethod turb_method>
__device__ void
reconstruction(real *pv, real *pv_l, real *pv_r, integer idx_shared, DZone *zone, DParameter *param);

template<MixtureModel mix_model, TurbMethod turb_method>
__device__ void
AUSMP_compute_inviscid_flux(DZone *zone, real *pv, integer tid, DParameter *param, real *fc, real *metric,
                            const real *jac);

// Implementations

template<MixtureModel mix_model, TurbMethod turb_method>
__device__ void
reconstruction(real *pv, real *pv_l, real *pv_r, const integer idx_shared, DZone *zone,
               DParameter *param) {
  const auto n_var = zone->n_var;
  switch (param->reconstruction) {
    case 2:MUSCL_reconstruct(pv, pv_l, pv_r, idx_shared, n_var, param->limiter);
      break;
    case 3:NND2_reconstruct(pv, pv_l, pv_r, idx_shared, n_var, param->limiter);
      break;
    default:first_order_reconstruct(pv, pv_l, pv_r, idx_shared, n_var);
  }
  if constexpr (mix_model != MixtureModel::Air) {
    real el = 0.5 * (pv_l[1] * pv_l[1] + pv_l[2] * pv_l[2] + pv_l[3] * pv_l[3]);
    real er = 0.5 * (pv_r[1] * pv_r[1] + pv_r[2] * pv_r[2] + pv_r[3] * pv_r[3]);
    const auto n_spec = zone->n_spec;
    real mw_inv_l{0.0}, mw_inv_r{0.0};
    for (int l = 0; l < n_spec; ++l) {
      mw_inv_l += pv_l[5 + l] / param->mw[l];
      mw_inv_r += pv_r[5 + l] / param->mw[l];
    }
    const real t_l = pv_l[4] / (pv_l[0] * R_u * mw_inv_l);
    const real t_r = pv_r[4] / (pv_r[0] * R_u * mw_inv_r);

    real hl[MAX_SPEC_NUMBER], hr[MAX_SPEC_NUMBER], cpl_i[MAX_SPEC_NUMBER], cpr_i[MAX_SPEC_NUMBER];
    compute_enthalpy_and_cp(t_l, hl, cpl_i, param);
    compute_enthalpy_and_cp(t_r, hr, cpr_i, param);
    real cpl{0}, cpr{0}, cvl{0}, cvr{0};
    for (auto l = 0; l < n_spec; ++l) {
      cpl += cpl_i[l] * pv_l[l + 5];
      cpr += cpr_i[l] * pv_r[l + 5];
      cvl += pv_l[l + 5] * (cpl_i[l] - R_u / param->mw[l]);
      cvr += pv_r[l + 5] * (cpr_i[l] - R_u / param->mw[l]);
      el += hl[l] * pv_l[l + 5];
      er += hr[l] * pv_r[l + 5];
    }
    pv_l[n_var] = pv_l[0] * el - pv_l[4]; //total energy
    pv_r[n_var] = pv_r[0] * er - pv_r[4];

    pv_l[n_var + 1] = cpl / cvl; //specific heat ratio
    pv_r[n_var + 1] = cpr / cvr;
  } else {
    real el = 0.5 * (pv_l[1] * pv_l[1] + pv_l[2] * pv_l[2] + pv_l[3] * pv_l[3]);
    real er = 0.5 * (pv_r[1] * pv_r[1] + pv_r[2] * pv_r[2] + pv_r[3] * pv_r[3]);
    pv_l[n_var] = el * pv_l[0] + pv_l[4] / (gamma_air - 1);
    pv_r[n_var] = er * pv_r[0] + pv_r[4] / (gamma_air - 1);
  }
}

template<MixtureModel mix_model, TurbMethod turb_method>
__device__ void
AUSMP_compute_inviscid_flux(DZone *zone, real *pv, integer tid, DParameter *param, real *fc, real *metric,
                            const real *jac) {
  const auto ng{zone->ngg};
  constexpr integer n_reconstruction = 7 + MAX_SPEC_NUMBER + 4; // rho,u,v,w,p,Y_{1...Ns},(k,omega,z,z_prime),E,gamma
  real pv_l[n_reconstruction], pv_r[n_reconstruction];
  const integer i_shared = tid - 1 + ng;
  reconstruction<mix_model, turb_method>(pv, pv_l, pv_r, i_shared, zone, param);

  auto metric_l = &metric[i_shared * 3], metric_r = &metric[(i_shared + 1) * 3];
  auto jac_l = jac[i_shared], jac_r = jac[i_shared + 1];
  const real k1 = 0.5 * (jac_l * metric_l[0] + jac_r * metric_r[0]);
  const real k2 = 0.5 * (jac_l * metric_l[1] + jac_r * metric_r[1]);
  const real k3 = 0.5 * (jac_l * metric_l[2] + jac_r * metric_r[2]);
  const real grad_k_div_jac = std::sqrt(k1 * k1 + k2 * k2 + k3 * k3);

  const real ul = (k1 * pv_l[1] + k2 * pv_l[2] + k3 * pv_l[3]) / grad_k_div_jac;
  const real ur = (k1 * pv_r[1] + k2 * pv_r[2] + k3 * pv_r[3]) / grad_k_div_jac;

  const real pl = pv_l[4], pr = pv_r[4], rho_l = pv_l[0], rho_r = pv_r[0];
  const integer n_spec = zone->n_spec;
  real gam_l{gamma_air}, gam_r{gamma_air};
  const integer n_var = zone->n_var;
  if (n_spec > 0) {
    gam_l = pv_l[n_var + 1];
    gam_r = pv_r[n_var + 1];
  }
  const real c = 0.5 * (std::sqrt(gam_l * pl / rho_l) + std::sqrt(gam_r * pr / rho_r));
  const real mach_l = ul / c, mach_r = ur / c;
  real mlp, mrn, plp, prn; // m for M, l/r for L/R, p/n for +/-. mlp=M_L^+
  constexpr static real alpha{3 / 16.0};
  if (std::abs(mach_l) > 1) {
    mlp = 0.5 * (mach_l + std::abs(mach_l));
    plp = mlp / mach_l;
  } else {
    const real ml_plus1_squared_div4 = (mach_l + 1) * (mach_l + 1) * 0.25;
    const real ml_squared_minus_1_squared = (mach_l * mach_l - 1) * (mach_l * mach_l - 1);
    mlp = ml_plus1_squared_div4 + 0.125 * ml_squared_minus_1_squared;
    plp = ml_plus1_squared_div4 * (2 - mach_l) + alpha * mach_l * ml_squared_minus_1_squared;
  }
  if (std::abs(mach_r) > 1) {
    mrn = 0.5 * (mach_r - std::abs(mach_r));
    prn = mrn / mach_r;
  } else {
    const real mr_minus1_squared_div4 = (mach_r - 1) * (mach_r - 1) * 0.25;
    const real mr_squared_minus_1_squared = (mach_r * mach_r - 1) * (mach_r * mach_r - 1);
    mrn = -mr_minus1_squared_div4 - 0.125 * mr_squared_minus_1_squared;
    prn = mr_minus1_squared_div4 * (2 + mach_r) - alpha * mach_r * mr_squared_minus_1_squared;
  }

  const real p_coeff = plp * pl + prn * pr;

  const real m_half = mlp + mrn;
  const real mach_pos = 0.5 * (m_half + std::abs(m_half));
  const real mach_neg = 0.5 * (m_half - std::abs(m_half));
  const real mass_flux_half = c * (rho_l * mach_pos + rho_r * mach_neg);
  const real coeff = mass_flux_half * grad_k_div_jac;

  auto fci = &fc[tid * n_var];
  if (mass_flux_half >= 0) {
    fci[0] = coeff;
    fci[1] = coeff * pv_l[1] + p_coeff * k1;
    fci[2] = coeff * pv_l[2] + p_coeff * k2;
    fci[3] = coeff * pv_l[3] + p_coeff * k3;
    fci[4] = coeff * (pv_l[n_var] + pv_l[4]) / pv_l[0];
    for (int l = 5; l < n_var; ++l) {
      fci[l] = coeff * pv_l[l];
    }
  } else {
    fci[0] = coeff;
    fci[1] = coeff * pv_r[1] + p_coeff * k1;
    fci[2] = coeff * pv_r[2] + p_coeff * k2;
    fci[3] = coeff * pv_r[3] + p_coeff * k3;
    fci[4] = coeff * (pv_r[n_var] + pv_r[4]) / pv_r[0];
    for (int l = 5; l < n_var; ++l) {
      fci[l] = coeff * pv_r[l];
    }
  }
}

} // cfd
