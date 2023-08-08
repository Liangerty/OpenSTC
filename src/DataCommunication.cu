#include "DataCommunication.cuh"
#include "Field.h"

__global__ void cfd::inner_communication(DZone *zone, DZone *tar_zone, integer i_face){
  const auto& f = zone->innerface[i_face];
  uint n[3];
  n[0] = blockIdx.x * blockDim.x + threadIdx.x;
  n[1] = blockDim.y * blockIdx.y + threadIdx.y;
  n[2] = blockIdx.z * blockDim.z + threadIdx.z;
  if (n[0] >= f.n_point[0] || n[1] >= f.n_point[1] || n[2] >= f.n_point[2]) return;

  integer idx[3], idx_tar[3], d_idx[3];
  for (integer i = 0; i < 3; ++i) {
    d_idx[i] = f.loop_dir[i] * (integer)(n[i]);
    idx[i] = f.range_start[i] + d_idx[i];
  }
  for (integer i = 0; i < 3; ++i) {
    idx_tar[i] = f.target_start[i] + f.target_loop_dir[i] * d_idx[f.src_tar[i]];
  }

  // The face direction: which of i(0)/j(1)/k(2) is the coincided face.
  const auto face_dir{f.direction > 0 ? f.range_start[f.face] : f.range_end[f.face]};

  if (idx[f.face] == face_dir) {
    // If this is the corresponding face, then average the values from both blocks
    for (integer l = 0; l < 6; ++l) {
      const real ave =
          0.5 * (tar_zone->bv(idx_tar[0], idx_tar[1], idx_tar[2], l) + zone->bv(idx[0], idx[1], idx[2], l));
      zone->bv(idx[0], idx[1], idx[2], l) = ave;
      tar_zone->bv(idx_tar[0], idx_tar[1], idx_tar[2], l) = ave;
    }
    for (int l = 0; l < zone->n_var; ++l) {
      const real ave =
          0.5 * (tar_zone->cv(idx_tar[0], idx_tar[1], idx_tar[2], l) + zone->cv(idx[0], idx[1], idx[2], l));
      zone->cv(idx[0], idx[1], idx[2], l) = ave;
      tar_zone->cv(idx_tar[0], idx_tar[1], idx_tar[2], l) = ave;
    }
    for (int l = 0; l < zone->n_scal; ++l) {
      // Be Careful! The flamelet case is different from here, should be pay extra attention!
      real ave = 0.5 * (tar_zone->sv(idx_tar[0], idx_tar[1], idx_tar[2], l) + zone->sv(idx[0], idx[1], idx[2], l));
      zone->sv(idx[0], idx[1], idx[2], l) = ave;
      tar_zone->sv(idx_tar[0], idx_tar[1], idx_tar[2], l) = ave;
    }
  } else {
    // Else, get the inner value for this block's ghost grid
    for (int l = 0; l < 5; ++l) {
      zone->bv(idx[0], idx[1], idx[2], l) = tar_zone->bv(idx_tar[0], idx_tar[1], idx_tar[2], l);
      zone->cv(idx[0], idx[1], idx[2], l) = tar_zone->cv(idx_tar[0], idx_tar[1], idx_tar[2], l);
    }
    zone->bv(idx[0], idx[1], idx[2], 5) = tar_zone->bv(idx_tar[0], idx_tar[1], idx_tar[2], 5);
    for (int l = 0; l < zone->n_scal; ++l) {
      // Be Careful! The flamelet case is different from here, should be pay extra attention!
      zone->sv(idx[0], idx[1], idx[2], l) = tar_zone->sv(idx_tar[0], idx_tar[1], idx_tar[2], l);
      zone->cv(idx[0], idx[1], idx[2], l + 5) = tar_zone->cv(idx_tar[0], idx_tar[1], idx_tar[2], l + 5);
    }
  }
}
