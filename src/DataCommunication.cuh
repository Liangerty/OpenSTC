#pragma once
#include "Define.h"
#include <vector>
#include "Mesh.h"

namespace cfd{
template<MixtureModel mix_model, TurbMethod turb_method>
struct Field;

template<MixtureModel mix_model, TurbMethod turb_method>
void data_communication(const Mesh &mesh, std::vector<cfd::Field<mix_model, turb_method>>& field);

struct DZone;
__global__ void inner_communication(DZone *zone, DZone *tar_zone, integer i_face);

template<MixtureModel mix_model, TurbMethod turb_method>
void data_communication(const Mesh &mesh, std::vector<cfd::Field<mix_model, turb_method>>& field) {
  // -1 - inner faces
  for (auto blk = 0; blk < mesh.n_block; ++blk) {
    auto &inF = mesh[blk].inner_face;
    const auto n_innFace = inF.size();
    auto v = field[blk].d_ptr;
    const auto ngg = mesh[blk].ngg;
    for (auto l = 0; l < n_innFace; ++l) {
      // reference to the current face
      const auto &fc = mesh[blk].inner_face[l];
      uint tpb[3], bpg[3];
      for (size_t j = 0; j < 3; ++j) {
        tpb[j] = fc.n_point[j] <= (2 * ngg + 1) ? 1 : 16;
        bpg[j] = (fc.n_point[j] - 1) / tpb[j] + 1;
      }
      dim3 TPB{tpb[0], tpb[1], tpb[2]}, BPG{bpg[0], bpg[1], bpg[2]};

      // variables of the neighbor block
      auto nv = field[fc.target_block].d_ptr;
      inner_communication<<<BPG, TPB>>>(v, nv, l);
    }
  }
}

}