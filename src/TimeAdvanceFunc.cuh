#pragma once
#include "Define.h"
#include "DParameter.h"
#include <cuda_runtime.h>
#include "Field.h"
#include "Constants.h"

namespace cfd {
struct DZone;

__global__ void store_last_step(DZone *zone);

__global__ void set_dq_to_0(cfd::DZone* zone);

template<MixtureModel mixture>
__global__ void local_time_step(cfd::DZone *zone, DParameter *param);

__global__ void compute_square_of_dbv(DZone *zone);
}

template<MixtureModel mixture>
__global__ void cfd::local_time_step(cfd::DZone *zone, cfd::DParameter *param) {
  const integer extent[3]{zone->mx, zone->my, zone->mz};
  const integer i=blockDim.x*blockIdx.x+threadIdx.x;
  const integer j=blockDim.y*blockIdx.y+threadIdx.y;
  const integer k=blockDim.z*blockIdx.z+threadIdx.z;
  if (i>=extent[0]||j>=extent[1]||k>=extent[2]) return;

  const auto& m{zone->metric(i, j, k)};
  const auto& bv=zone->bv;
  const integer dim{zone->mz == 1 ? 2 : 3};

  const real grad_xi = std::sqrt(m(1, 1) * m(1, 1) + m(1, 2) * m(1, 2) + m(1, 3) * m(1, 3));
  const real grad_eta = std::sqrt(m(2, 1) * m(2, 1) + m(2, 2) * m(2, 2) + m(2, 3) * m(2, 3));
  const real grad_zeta = std::sqrt(m(3, 1) * m(3, 1) + m(3, 2) * m(3, 2) + m(3, 3) * m(3, 3));

  const real u{bv(i, j, k, 1)}, v{bv(i, j, k, 2)}, w{bv(i, j, k, 3)};
  const real U = u * m(1, 1) + v * m(1, 2) + w * m(1, 3);
  const real V = u * m(2, 1) + v * m(2, 2) + w * m(2, 3);
  const real W = u * m(3, 1) + v * m(3, 2) + w * m(3, 3);

  const auto acoustic_speed = zone->acoustic_speed(i, j, k);
  real spectral_radius_inviscid = std::abs(U) + std::abs(V) + acoustic_speed * (grad_xi + grad_eta);
  if (dim == 3)
    spectral_radius_inviscid += std::abs(W) + acoustic_speed * grad_zeta;

  // Next, compute the viscous spectral radius
  real gamma{gamma_air};
  if constexpr (mixture!=MixtureModel::Air){
    gamma=zone->gamma(i,j,k);
  }
  real coeff_1 = max(gamma, 4.0 / 3.0);
  const real coeff_2 = zone->mul(i, j, k) / bv(i, j, k, 0) / param->Pr;
  real spectral_radius_viscous = grad_xi * grad_xi + grad_eta * grad_eta;
  if (dim == 3)
    spectral_radius_viscous += grad_zeta * grad_zeta;
  spectral_radius_viscous *= coeff_1 * coeff_2;

  zone->dt_local(i, j, k) = param->cfl / (spectral_radius_inviscid + spectral_radius_viscous);
}
