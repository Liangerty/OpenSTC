#pragma once

#include "InviscidScheme.cuh"
#include "ViscousScheme.cuh"

namespace cfd {
template<MixtureModel mix_model, TurbMethod turb_method>
void
compute_inviscid_flux(const Block &block, cfd::DZone *zone, DParameter *param, integer n_var);

template<MixtureModel mix_model, TurbMethod turb_method>
__global__ void
inviscid_flux_1d(cfd::DZone *zone, integer direction, integer max_extent, cfd::DParameter *param);

template<MixtureModel mix_model, TurbMethod turb_method>
void compute_viscous_flux(const Block &block, cfd::DZone *zone, DParameter *param, integer n_var);

template<MixtureModel mix_model, TurbMethod turb_method>
__global__ void
viscous_flux_fv(cfd::DZone *zone, integer max_extent, cfd::DParameter *param);

template<MixtureModel mix_model, TurbMethod turb_method>
__global__ void
viscous_flux_gv(cfd::DZone *zone, integer max_extent, cfd::DParameter *param);

template<MixtureModel mix_model, TurbMethod turb_method>
__global__ void
viscous_flux_hv(cfd::DZone *zone, integer max_extent, cfd::DParameter *param);

template<MixtureModel mix_model, TurbMethod turb_method>
__device__ void select_inviscid_scheme(DZone *zone, real *pv, integer tid, DParameter *param, real *fc, real *metric,
                                       const real *jac);

// Implementations

template<MixtureModel mix_model, TurbMethod turb_method>
void
compute_inviscid_flux(const Block &block, cfd::DZone *zone, DParameter *param, const integer n_var) {
  const integer extent[3]{block.mx, block.my, block.mz};
  constexpr integer block_dim = 128;
  const integer n_computation_per_block = block_dim + 2 * block.ngg - 1;
  const auto shared_mem = (block_dim * n_var // fc
                           + n_computation_per_block * (n_var + 3 + 1)) * sizeof(real); // pv[n_var]+metric[3]+jacobian

  for (auto dir = 0; dir < 2; ++dir) {
    integer tpb[3]{1, 1, 1};
    tpb[dir] = block_dim;
    integer bpg[3]{extent[0], extent[1], extent[2]};
    bpg[dir] = (extent[dir] - 1) / (tpb[dir] - 1) + 1;

    dim3 TPB(tpb[0], tpb[1], tpb[2]);
    dim3 BPG(bpg[0], bpg[1], bpg[2]);
    inviscid_flux_1d<mix_model, turb_method><<<BPG, TPB, shared_mem>>>(zone, dir, extent[dir], param);
  }

  if (extent[2] > 1) {
    // 3D computation
    // Number of threads in the 3rd direction cannot exceed 64
    integer tpb[3]{1, 1, 1};
    tpb[2] = 64;
    integer bpg[3]{extent[0], extent[1], extent[2]};
    bpg[2] = (extent[2] - 1) / (tpb[2] - 1) + 1;

    dim3 TPB(tpb[0], tpb[1], tpb[2]);
    dim3 BPG(bpg[0], bpg[1], bpg[2]);
    inviscid_flux_1d<mix_model, turb_method><<<BPG, TPB, shared_mem>>>(zone, 2, extent[2], param);
  }
}

template<MixtureModel mix_model, TurbMethod turb_method>
__global__ void
inviscid_flux_1d(cfd::DZone *zone, integer direction, integer max_extent, DParameter *param) {
  integer labels[3]{0, 0, 0};
  labels[direction] = 1;
  const integer tid = threadIdx.x * labels[0] + threadIdx.y * labels[1] + threadIdx.z * labels[2];
  const integer block_dim = blockDim.x * blockDim.y * blockDim.z;
  const auto ngg{zone->ngg};
  const integer n_point = block_dim + 2 * ngg - 1;

  integer idx[3];
  idx[0] = (integer) ((blockDim.x - labels[0]) * blockIdx.x + threadIdx.x);
  idx[1] = (integer) ((blockDim.y - labels[1]) * blockIdx.y + threadIdx.y);
  idx[2] = (integer) ((blockDim.z - labels[2]) * blockIdx.z + threadIdx.z);
  idx[direction] -= 1;
  if (idx[direction] >= max_extent) return;

  // load variables to shared memory
  extern __shared__ real s[];
  const auto n_var{zone->n_var};
  real *pv = s;
  real *metric = &pv[n_point * n_var];
  real *jac = &metric[n_point * 3];
  real *fc = &jac[n_point];

  const integer i_shared = tid - 1 + ngg;
  for (auto l = 0; l < 5; ++l) { // 0-rho,1-u,2-v,3-w,4-p
    pv[i_shared * n_var + l] = zone->bv(idx[0], idx[1], idx[2], l);
  }
  const auto n_scalar{zone->n_scal};
  for (auto l = 0; l < n_scalar; ++l) { // Y_k, k, omega, Z, Z_prime...
    pv[i_shared * n_var + 5 + l] = zone->sv(idx[0], idx[1], idx[2], l);
  }
  for (auto l = 1; l < 4; ++l) {
    metric[i_shared * 3 + l - 1] = zone->metric(idx[0], idx[1], idx[2])(direction + 1, l);
  }
  jac[i_shared] = zone->jac(idx[0], idx[1], idx[2]);

  // ghost cells
  if (tid == 0) {
    // Responsible for the left (ngg-1) points
    for (auto i = 1; i < ngg; ++i) {
      const auto ig_shared = ngg - 1 - i;
      const integer g_idx[3]{idx[0] - i * labels[0], idx[1] - i * labels[1], idx[2] - i * labels[2]};

      for (auto l = 0; l < 5; ++l) { // 0-rho,1-u,2-v,3-w,4-p
        pv[ig_shared * n_var + l] = zone->bv(g_idx[0], g_idx[1], g_idx[2], l);
      }
      for (auto l = 0; l < n_scalar; ++l) { // Y_k, k, omega, Z, Z_prime...
        pv[ig_shared * n_var + 5 + l] = zone->sv(g_idx[0], g_idx[1], g_idx[2], l);
      }
      for (auto l = 1; l < 4; ++l) {
        metric[ig_shared * 3 + l - 1] = zone->metric(g_idx[0], g_idx[1], g_idx[2])(direction + 1, l);
      }
      jac[ig_shared] = zone->jac(g_idx[0], g_idx[1], g_idx[2]);
    }
  }
  if (tid == block_dim - 1 || idx[direction] == max_extent - 1) {
    // Responsible for the right ngg points
    for (auto i = 1; i <= ngg; ++i) {
      const auto ig_shared = tid + i + ngg - 1;
      const integer g_idx[3]{idx[0] + i * labels[0], idx[1] + i * labels[1], idx[2] + i * labels[2]};

      for (auto l = 0; l < 5; ++l) { // 0-rho,1-u,2-v,3-w,4-p
        pv[ig_shared * n_var + l] = zone->bv(g_idx[0], g_idx[1], g_idx[2], l);
      }
      for (auto l = 0; l < n_scalar; ++l) { // Y_k, k, omega, Z, Z_prime...
        pv[ig_shared * n_var + 5 + l] = zone->sv(g_idx[0], g_idx[1], g_idx[2], l);
      }
      for (auto l = 1; l < 4; ++l) {
        metric[ig_shared * 3 + l - 1] = zone->metric(g_idx[0], g_idx[1], g_idx[2])(direction + 1, l);
      }
      jac[ig_shared] = zone->jac(g_idx[0], g_idx[1], g_idx[2]);
    }
  }
  __syncthreads();

//  (*inviscid_scheme)->compute_inviscid_flux(zone, pv, tid, param, fc, metric, jac);
  select_inviscid_scheme<mix_model, turb_method>(zone, pv, tid, param, fc, metric, jac);
  __syncthreads();


  if (tid > 0) {
    for (integer l = 0; l < n_var; ++l) {
      zone->dq(idx[0], idx[1], idx[2], l) -= fc[tid * n_var + l] - fc[(tid - 1) * n_var + l];
    }
  }
}

template<MixtureModel mix_model, TurbMethod turb_method>
__device__ void
select_inviscid_scheme(DZone *zone, real *pv, integer tid, DParameter *param, real *fc, real *metric,
                       const real *jac) {
  const integer inviscid_scheme = param->inviscid_scheme;
  switch (inviscid_scheme) {
    case 2: // Roe
    case 3: // AUSM+
    default:AUSMP_compute_inviscid_flux<mix_model, turb_method>(zone, pv, tid, param, fc, metric, jac);
      break;
  }
}

template<MixtureModel mix_model, TurbMethod turb_method>
void compute_viscous_flux(const Block &block, cfd::DZone *zone, DParameter *param, integer n_var) {
  const integer extent[3]{block.mx, block.my, block.mz};
  const integer dim{extent[2] == 1 ? 2 : 3};
  constexpr integer block_dim = 64;

  dim3 tpb{block_dim, 1, 1};
  dim3 bpg((extent[0] - 1) / (block_dim - 1) + 1, extent[1], extent[2]);
  auto shared_mem = block_dim * n_var * sizeof(real);
  viscous_flux_fv<mix_model, turb_method><<<bpg, tpb, shared_mem>>>(zone, extent[0], param);

  tpb = {1, block_dim, 1};
  bpg = dim3(extent[0], (extent[1] - 1) / (block_dim - 1) + 1, extent[2]);
  viscous_flux_gv<mix_model, turb_method><<<bpg, tpb, shared_mem>>>(zone, extent[1], param);

  if (dim == 3) {
    tpb = {1, 1, block_dim};
    bpg = dim3(extent[0], extent[1], (extent[2] - 1) / (block_dim - 1) + 1);
    viscous_flux_hv<mix_model, turb_method><<<bpg, tpb, shared_mem>>>(zone, extent[2], param);
  }
}

template<MixtureModel mix_model, TurbMethod turb_method>
__global__ void
viscous_flux_fv(cfd::DZone *zone, integer max_extent, cfd::DParameter *param) {
  integer idx[3];
  idx[0] = ((integer) blockDim.x - 1) * blockIdx.x + threadIdx.x - 1;
  idx[1] = (integer) (blockDim.y * blockIdx.y + threadIdx.y);
  idx[2] = (integer) (blockDim.z * blockIdx.z + threadIdx.z);
  if (idx[0] >= max_extent) return;
  const auto tid = threadIdx.x;
  const auto n_var{zone->n_var};

  extern __shared__ real s[];
  real *fv = s;

  switch (param->viscous_scheme) {
    case 0: // Inviscid computation
      break;
    case 2:
    default: // 2nd order central difference
      compute_fv_2nd_order<mix_model, turb_method>(idx, zone, &fv[tid * n_var], param);
  }
  __syncthreads();

  if (tid > 0) {
    for (integer l = 0; l < n_var; ++l) {
      zone->dq(idx[0], idx[1], idx[2], l) += fv[tid * n_var + l] - fv[(tid - 1) * n_var + l];
    }
  }
}

template<MixtureModel mix_model, TurbMethod turb_method>
__global__ void viscous_flux_gv(cfd::DZone *zone, integer max_extent, cfd::DParameter *param) {
  integer idx[3];
  idx[0] = (integer) (blockDim.x * blockIdx.x + threadIdx.x);
  idx[1] = (integer) ((blockDim.y - 1) * blockIdx.y + threadIdx.y) - 1;
  idx[2] = (integer) (blockDim.z * blockIdx.z + threadIdx.z);
  if (idx[1] >= max_extent) return;
  const auto tid = threadIdx.y;
  const auto n_var{zone->n_var};

  extern __shared__ real s[];
  real *gv = s;

  switch (param->viscous_scheme) {
    case 0: // Inviscid computation
      break;
    case 2:
    default: // 2nd order central difference
      compute_gv_2nd_order<mix_model, turb_method>(idx, zone, &gv[tid * n_var], param);
  }
  __syncthreads();

  if (tid > 0) {
    for (integer l = 0; l < n_var; ++l) {
      zone->dq(idx[0], idx[1], idx[2], l) += gv[tid * n_var + l] - gv[(tid - 1) * n_var + l];
    }
  }
}

template<MixtureModel mix_model, TurbMethod turb_method>
__global__ void viscous_flux_hv(cfd::DZone *zone, integer max_extent, cfd::DParameter *param) {
  integer idx[3];
  idx[0] = (integer) (blockDim.x * blockIdx.x + threadIdx.x);
  idx[1] = (integer) (blockDim.y * blockIdx.y + threadIdx.y);
  idx[2] = (integer) ((blockDim.z - 1) * blockIdx.z + threadIdx.z) - 1;
  if (idx[2] >= max_extent) return;
  const auto tid = threadIdx.z;
  const auto n_var{zone->n_var};

  extern __shared__ real s[];
  real *hv = s;

  switch (param->viscous_scheme) {
    case 0: // Inviscid computation
      break;
    case 2:
    default: // 2nd order central difference
      compute_hv_2nd_order<mix_model, turb_method>(idx, zone, &hv[tid * n_var], param);
  }
  __syncthreads();

  if (tid > 0) {
    for (integer l = 0; l < n_var; ++l) {
      zone->dq(idx[0], idx[1], idx[2], l) += hv[tid * n_var + l] - hv[(tid - 1) * n_var + l];
    }
  }
}

}