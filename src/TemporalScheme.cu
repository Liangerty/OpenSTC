#include "TemporalScheme.cuh"
#include "DParameter.h"
#include "Field.h"
#if MULTISPECIES==1
#else
#include "Constants.h"
#endif

namespace cfd {
__device__ void
SteadyTemporalScheme::compute_time_step(DZone *zone, integer i, integer j, integer k, const DParameter *param) {
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
#if MULTISPECIES==1
  const real coeff_1 = max(zone->gamma(i, j, k), 4.0 / 3.0);
#else
  const real coeff_1 = max(gamma_air, 4.0 / 3.0);
#endif
  const real coeff_2 = zone->mul(i, j, k) / bv(i, j, k, 0) / param->Pr;
  real spectral_radius_viscous = grad_xi * grad_xi + grad_eta * grad_eta;
  if (dim == 3)
    spectral_radius_viscous += grad_zeta * grad_zeta;
  spectral_radius_viscous *= coeff_1 * coeff_2;

  zone->dt_local(i, j, k) = param->cfl / (spectral_radius_inviscid + spectral_radius_viscous);
}

//SteadyTemporalScheme::SteadyTemporalScheme(const Parameter &parameter, const Mesh &mesh) {
//  std::vector<ggxl::Array3D<real>> h_dt;
//  for (integer i = 0; i < mesh.n_block; ++i) {
//    const integer mx{mesh[i].mx}, my{mesh[i].my}, mz{mesh[i].mz};
//    h_dt.emplace_back(mx, my, mz);
//  }
//  const auto mem_sz=mesh.n_block* sizeof(ggxl::Array3D<real>);
//  cudaMalloc(&dt, mem_sz);
//  cudaMemcpy(dt,h_dt.data(),mem_sz,cudaMemcpyHostToDevice);
//}
//
//LUSGS::LUSGS(const Parameter &parameter, const Mesh &mesh) : SteadyTemporalScheme(parameter, mesh) {
//  std::vector<ggxl::Array3D<real[3]>> h_inv_rad;
//  std::vector<ggxl::Array3D<real>> h_vis_rad;
//  for (integer i = 0; i < mesh.n_block; ++i) {
//    const integer mx{mesh[i].mx}, my{mesh[i].my}, mz{mesh[i].mz};
//    h_inv_rad.emplace_back(mx, my, mz);
//    h_vis_rad.emplace_back(mx, my, mz);
//  }
//  auto mem_sz=mesh.n_block* sizeof(ggxl::Array3D<real[3]>);
//  cudaMalloc(&inviscid_spectral_radius, mem_sz);
//  cudaMemcpy(inviscid_spectral_radius,h_inv_rad.data(),mem_sz,cudaMemcpyHostToDevice);
//  mem_sz=mesh.n_block* sizeof(ggxl::Array3D<real>);
//  cudaMalloc(&viscous_spectral_radius, mem_sz);
//  cudaMemcpy(viscous_spectral_radius,h_vis_rad.data(),mem_sz,cudaMemcpyHostToDevice);
//}
} // cfd