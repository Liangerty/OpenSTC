#pragma once

#include "Define.h"
#include "Mesh.h"
#include "Field.h"
#include "DParameter.h"
#include "SST.cuh"
#include "FiniteRateChem.cuh"

namespace cfd {
template<MixtureModel mix_model, TurbMethod turb_method>
__global__ void compute_source(cfd::DZone *zone, DParameter *param) {
  const integer extent[3]{zone->mx, zone->my, zone->mz};
  const auto i = (integer) (blockDim.x * blockIdx.x + threadIdx.x);
  const auto j = (integer) (blockDim.y * blockIdx.y + threadIdx.y);
  const auto k = (integer) (blockDim.z * blockIdx.z + threadIdx.z);
  if (i >= extent[0] || j >= extent[1] || k >= extent[2]) return;

  if constexpr (turb_method == TurbMethod::RANS) {
    switch (param->rans_model) {
      case 1://SA
        break;
      case 2:
      default: // SST
        SST::compute_source_and_mut(zone, i, j, k, param);
    }
    // The mut is always computed in above functions, and we compute turbulent thermal conductivity here
    if constexpr (mix_model != MixtureModel::Air) {
      zone->turb_therm_cond(i, j, k) = zone->mut(i, j, k) * zone->cp(i, j, k) / param->Prt;
    } else {
      constexpr real cp{gamma_air * R_u / mw_air / (gamma_air - 1)};
      zone->turb_therm_cond(i, j, k) = zone->mut(i, j, k) * cp / param->Prt;
    }
  }

  if constexpr (mix_model == MixtureModel::FR) {
    // Finite rate chemistry will be computed
    finite_rate_chemistry(zone, i, j, k, param);
  }
}
}