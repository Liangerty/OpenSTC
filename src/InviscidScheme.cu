#include "InviscidScheme.cuh"
#include "DParameter.h"
#include "Field.h"
#if MULTISPECIES==1
#else
#include "Constants.h"
#endif

namespace cfd {
__device__ InviscidScheme::InviscidScheme(DParameter *param) {
  integer reconstruct_tag = param->reconstruction;
  switch (reconstruct_tag) {
    case 2:reconstruction_method = new MUSCL(param);
      break;
    default:reconstruction_method = new Reconstruction(param);
      break;
  }
}

__device__ AUSMP::AUSMP(DParameter *param) : InviscidScheme(param) {

}

__device__ void
AUSMP::compute_inviscid_flux(DZone *zone, real *pv, const integer tid, DParameter *param, real *fc, real *metric,
                             real *jac) {
  const auto ng{zone->ngg};
#if MULTISPECIES == 1
  constexpr integer n_reconstruction = 7+MAX_SPEC_NUMBER; // rho,u,v,w,p,Y_{1...Ns},E,gamma
#else
  constexpr integer n_reconstruction=6; // rho,u,v,w,p,E
#endif
  real pv_l[n_reconstruction], pv_r[n_reconstruction];
  const integer i_shared = tid - 1 + ng;
  reconstruction(pv, pv_l, pv_r, reconstruction_method, i_shared, zone, param);

  auto metr_l = &metric[i_shared * 3], metr_r = &metric[(i_shared + 1) * 3];
  auto jac_l = jac[i_shared], jac_r = jac[i_shared + 1];
  const real k1 = 0.5 * (jac_l * metr_l[0] + jac_r * metr_r[0]);
  const real k2 = 0.5 * (jac_l * metr_l[1] + jac_r * metr_r[1]);
  const real k3 = 0.5 * (jac_l * metr_l[2] + jac_r * metr_r[2]);
  const real grad_k_div_jac = std::sqrt(k1 * k1 + k2 * k2 + k3 * k3);

  const real ul = (k1 * pv_l[1] + k2 * pv_l[2] + k3 * pv_l[3]) / grad_k_div_jac;
  const real ur = (k1 * pv_r[1] + k2 * pv_r[2] + k3 * pv_r[3]) / grad_k_div_jac;

  const real pl = pv_l[4], pr = pv_r[4], rho_l = pv_l[0], rho_r = pv_r[0];
  const integer n_spec=zone->n_spec;
#if MULTISPECIES == 1
  const real gam_l = pv_l[6 + n_spec], gam_r = pv_r[6 + n_spec];
  const real c = 0.5 * (std::sqrt(gam_l * pl / rho_l) + std::sqrt(gam_r * pr / rho_r));
#else
  const real c  = 0.5 * (std::sqrt(gamma_air*pl/rho_l) + std::sqrt(gamma_air * pr / rho_r));
#endif
  const real mach_l = ul / c, mach_r = ur / c;
  real mlp{0}, mrn{0}, plp{0}, prn{0}; // m for M, l/r for L/R, p/n for +/-. mlp=M_L^+
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

  integer n_var = zone->n_var;
  auto fci = &fc[tid*n_var];
  if (mass_flux_half >= 0) {
    fci[0] = coeff;
    fci[1] = coeff * pv_l[1] + p_coeff * k1;
    fci[2] = coeff * pv_l[2] + p_coeff * k2;
    fci[3] = coeff * pv_l[3] + p_coeff * k3;
    fci[4] = coeff * (pv_l[5 + n_spec] + pv_l[4]) / pv_l[0];
    for (int l = 5; l < n_var; ++l) {
      fci[l] = coeff * pv_l[l];
    }
  } else {
    fci[0] = coeff;
    fci[1] = coeff * pv_r[1] + p_coeff * k1;
    fci[2] = coeff * pv_r[2] + p_coeff * k2;
    fci[3] = coeff * pv_r[3] + p_coeff * k3;
    fci[4] = coeff * (pv_r[5 + n_spec] + pv_r[4]) / pv_r[0];
    for (int l = 5; l < n_var; ++l) {
      fci[l] = coeff * pv_r[l];
    }
  }
}
} // cfd