#include "TimeAdvanceFunc.cuh"
#include "Field.h"
#include "Thermo.cuh"

__global__ void cfd::store_last_step(cfd::DZone *zone) {
  const integer mx{zone->mx}, my{zone->my}, mz{zone->mz};
  integer i = blockDim.x * blockIdx.x + threadIdx.x;
  integer j = blockDim.y * blockIdx.y + threadIdx.y;
  integer k = blockDim.z * blockIdx.z + threadIdx.z;
  if (i >= mx || j >= my || k >= mz) return;

  zone->bv_last(i, j, k, 0) = zone->bv(i, j, k, 0);
  zone->bv_last(i, j, k, 1) = zone->vel(i, j, k);
  zone->bv_last(i, j, k, 2) = zone->bv(i, j, k, 4);
  zone->bv_last(i, j, k, 3) = zone->bv(i, j, k, 5);
}

__global__ void cfd::compute_square_of_dbv(cfd::DZone *zone) {
  const integer mx{zone->mx}, my{zone->my}, mz{zone->mz};
  integer i = blockDim.x * blockIdx.x + threadIdx.x;
  integer j = blockDim.y * blockIdx.y + threadIdx.y;
  integer k = blockDim.z * blockIdx.z + threadIdx.z;
  if (i >= mx || j >= my || k >= mz) return;

  auto &bv = zone->bv;
  auto &bv_last = zone->bv_last;

  bv_last(i, j, k, 0) = (bv(i, j, k, 0) - bv_last(i, j, k, 0)) * (bv(i, j, k, 0) - bv_last(i, j, k, 0));
  bv_last(i, j, k, 1) = (zone->vel(i, j, k) - bv_last(i, j, k, 1)) * (zone->vel(i, j, k) - bv_last(i, j, k, 1));
  bv_last(i, j, k, 2) = (bv(i, j, k, 4) - bv_last(i, j, k, 2)) * (bv(i, j, k, 4) - bv_last(i, j, k, 2));
  bv_last(i, j, k, 3) = (bv(i, j, k, 5) - bv_last(i, j, k, 3)) * (bv(i, j, k, 5) - bv_last(i, j, k, 3));
}

__global__ void cfd::limit_flow(cfd::DZone *zone, cfd::DParameter *param, integer blk_id) {
  const integer mx{zone->mx}, my{zone->my}, mz{zone->mz};
  const integer i = blockDim.x * blockIdx.x + threadIdx.x;
  const integer j = blockDim.y * blockIdx.y + threadIdx.y;
  const integer k = blockDim.z * blockIdx.z + threadIdx.z;
  if (i >= mx || j >= my || k >= mz) return;

  auto &bv = zone->bv;
  auto &sv = zone->sv;

  // Record the computed values. First for flow variables and mass fractions
  constexpr integer n_flow_var = 5;
  real var[n_flow_var];
  var[0] = bv(i, j, k, 0);
  var[1] = bv(i, j, k, 1);
  var[2] = bv(i, j, k, 2);
  var[3] = bv(i, j, k, 3);
  var[4] = bv(i, j, k, 4);
  const integer n_spec{zone->n_spec};

  // Find the unphysical values and limit them
  auto ll = param->limit_flow.ll;
  auto ul = param->limit_flow.ul;
  bool unphysical{false};
  for (integer l = 0; l < n_flow_var; ++l) {
    if (isnan(var[l])) {
      unphysical = true;
      break;
    }
    if (var[l] < ll[l] || var[l] > ul[l]) {
      unphysical = true;
      break;
    }
  }

  if (unphysical) {
    // printf("Unphysical values appear in process %d, block %d, i = %d, j = %d, k = %d.\n", param->myid, blk_id, i, j, k);

    real updated_var[n_flow_var + MAX_SPEC_NUMBER];
    memset(updated_var, 0, (n_flow_var + MAX_SPEC_NUMBER) * sizeof(real));
    integer kn{0};
    // Compute the sum of all "good" points surrounding the "bad" point
    for (integer ka = -1; ka < 2; ++ka) {
      const integer k1{k + ka};
      if (k1 < 0 || k1 >= mz) continue;
      for (integer ja = -1; ja < 2; ++ja) {
        const integer j1{j + ja};
        if (j1 < 0 || j1 >= my) continue;
        for (integer ia = -1; ia < 2; ++ia) {
          const integer i1{i + ia};
          if (i1 < 0 || i1 >= mz)continue;

          if (isnan(bv(i1, j1, k1, 0)) || isnan(bv(i1, j1, k1, 1)) || isnan(bv(i1, j1, k1, 2)) ||
              isnan(bv(i1, j1, k1, 3)) || isnan(bv(i1, j1, k1, 4)) || bv(i1, j1, k1, 0) < ll[0] ||
              bv(i1, j1, k1, 1) < ll[1] || bv(i1, j1, k1, 2) < ll[2] || bv(i1, j1, k1, 3) < ll[3] ||
              bv(i1, j1, k1, 4) < ll[4] || bv(i1, j1, k1, 0) > ul[0] || bv(i1, j1, k1, 1) > ul[1] ||
              bv(i1, j1, k1, 2) > ul[2] || bv(i1, j1, k1, 3) > ul[3] || bv(i1, j1, k1, 4) > ul[4]) {
            continue;
          }

          updated_var[0] += bv(i1, j1, k1, 0);
          updated_var[1] += bv(i1, j1, k1, 1);
          updated_var[2] += bv(i1, j1, k1, 2);
          updated_var[3] += bv(i1, j1, k1, 3);
          updated_var[4] += bv(i1, j1, k1, 4);

          for (integer l = 0; l < n_spec; ++l) {
            updated_var[l + 5] += sv(i1, j1, k1, l);
          }

          ++kn;
        }
      }
    }

    // Compute the average of the surrounding points
    if (kn > 0) {
      const real kn_inv{1.0 / kn};
      for (integer l = 0; l < n_flow_var + n_spec; ++l) {
        updated_var[l] *= kn_inv;
      }
    } else {
      // The surrounding points are all "bad"
      for (integer l = 0; l < 5; ++l) {
        updated_var[l] = max(var[l], ll[l]);
        updated_var[l] = min(updated_var[l], ul[l]);
      }
      for (integer l = 0; l < n_spec; ++l) {
        updated_var[l + 5] = param->limit_flow.sv_inf[l];
      }
    }

    // Assign averaged values for the bad point
    auto &cv = zone->cv;
    bv(i, j, k, 0) = updated_var[0];
    bv(i, j, k, 1) = updated_var[1];
    bv(i, j, k, 2) = updated_var[2];
    bv(i, j, k, 3) = updated_var[3];
    bv(i, j, k, 4) = updated_var[4];
    cv(i, j, k, 0) = updated_var[0];
    cv(i, j, k, 1) = updated_var[0] * updated_var[1];
    cv(i, j, k, 2) = updated_var[0] * updated_var[2];
    cv(i, j, k, 3) = updated_var[0] * updated_var[3];
    cv(i, j, k, 4) = 0.5 * updated_var[0] * (updated_var[1] * updated_var[1] + updated_var[2] * updated_var[2] +
                                             updated_var[3] * updated_var[3]);
    for (integer l = 0; l < n_spec; ++l) {
      sv(i, j, k, l) = updated_var[5 + l];
      cv(i, j, k, 5 + l) = updated_var[0] * updated_var[5 + l];
    }
    if (n_spec > 0) {
      real mw = 0;
      for (integer l = 0; l < n_spec; ++l) {
        mw += sv(i, j, k, l) / param->mw[l];
      }
      mw = 1 / mw;
      bv(i, j, k, 5) = updated_var[4] * mw / (updated_var[0] * R_u);
      real enthalpy[MAX_SPEC_NUMBER];
      compute_enthalpy(bv(i, j, k, 5), enthalpy, param);
      // Add species enthalpy together up to kinetic energy to get total enthalpy
      for (auto l = 0; l < zone->n_spec; l++) {
        cv(i, j, k, 4) += enthalpy[l] * cv(i, j, k, 5 + l);
      }
      cv(i, j, k, 4) -= bv(i, j, k, 4);  // (\rho e =\rho h - p)
    } else {
      bv(i, j, k, 5) = updated_var[4] * mw_air / (updated_var[0] * R_u);
      cv(i, j, k, 4) += updated_var[4] / (gamma_air - 1);
    }
  }

  // Limit the turbulent values
  if (param->rans_model == 2) {
    // Record the computed values
    constexpr integer n_turb = 2;
    real t_var[n_turb];
    t_var[0] = sv(i, j, k, n_spec);
    t_var[1] = sv(i, j, k, n_spec + 1);

    // Find the unphysical values and limit them
    unphysical = false;
    if (isnan(t_var[0]) || isnan(t_var[1]) || t_var[0] < 0 || t_var[1] < 0) {
      unphysical = true;
    }

    if (unphysical) {
      // printf("Unphysical turbulent values appear in process %d, block %d, i = %d, j = %d, k = %d.\n", param->myid,
      //        blk_id, i, j, k);

      real updated_var[n_turb];
      memset(updated_var, 0, n_turb * sizeof(real));
      integer kn{0};
      // Compute the sum of all "good" points surrounding the "bad" point
      for (integer ka = -1; ka < 2; ++ka) {
        const integer k1{k + ka};
        if (k1 < 0 || k1 >= mz) continue;
        for (integer ja = -1; ja < 2; ++ja) {
          const integer j1{j + ja};
          if (j1 < 0 || j1 >= my) continue;
          for (integer ia = -1; ia < 2; ++ia) {
            const integer i1{i + ia};
            if (i1 < 0 || i1 >= mz)continue;

            if (isnan(sv(i1, j1, k1, n_spec)) || isnan(sv(i1, j1, k1, 1 + n_spec)) || sv(i1, j1, k1, n_spec) < 0 ||
                sv(i1, j1, k1, n_spec + 1) < 0) {
              continue;
            }

            updated_var[0] += sv(i1, j1, k1, n_spec);
            updated_var[1] += sv(i1, j1, k1, 1 + n_spec);

            ++kn;
          }
        }
      }

      // Compute the average of the surrounding points
      if (kn > 0) {
        const real kn_inv{1.0 / kn};
        updated_var[0] *= kn_inv;
        updated_var[1] *= kn_inv;
      } else {
        // The surrounding points are all "bad"
        updated_var[0] = t_var[0] < 0 ? param->limit_flow.sv_inf[n_spec] : t_var[0];
        updated_var[1] = t_var[1] < 0 ? param->limit_flow.sv_inf[n_spec + 1] : t_var[1];
      }

      // Assign averaged values for the bad point
      auto &cv = zone->cv;
      sv(i, j, k, n_spec) = updated_var[0];
      sv(i, j, k, n_spec + 1) = updated_var[1];
      cv(i, j, k, n_spec + 5) = cv(i, j, k, 0) * updated_var[0];
      cv(i, j, k, n_spec + 6) = cv(i, j, k, 0) * updated_var[1];
    }
  }
}

