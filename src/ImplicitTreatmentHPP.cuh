#pragma once

#include "Define.h"
//#include "DParameter.h"
#include "DPLURHPP.cuh"
//#include "SST.cuh"
//#include "FiniteRateChem.cuh"

namespace cfd {
template<MixtureModel mixture_model, TurbMethod turb_method>
void implicit_treatment(const Block &block, const DParameter *param, DZone *d_ptr, const Parameter &parameter,
                        DZone *h_ptr) {
  switch (parameter.get_int("implicit_method")) {
    case 0: // Explicit
      if constexpr (mixture_model == MixtureModel::FR) {
        if (const integer chem_src_method = parameter.get_int("chemSrcMethod");chem_src_method != 0) {
          const integer extent[3]{block.mx, block.my, block.mz};
          const integer dim{extent[2] == 1 ? 2 : 3};
          dim3 tpb{8, 8, 4};
          if (dim == 2) {
            tpb = {16, 16, 1};
          }
          const dim3 bpg{(extent[0] - 1) / tpb.x + 1, (extent[1] - 1) / tpb.y + 1, (extent[2] - 1) / tpb.z + 1};
          switch (chem_src_method) {
            case 1: // EPI
              EPI<<<bpg, tpb>>>(d_ptr);
              break;
            case 2: // DA
              DA<<<bpg, tpb>>>(d_ptr);
              break;
            default: // explicit
              break;
          }
        }
      }
      if constexpr (turb_method == TurbMethod::RANS) {
        if (parameter.get_int("turb_implicit") == 1) {
          const integer extent[3]{block.mx, block.my, block.mz};
          const integer dim{extent[2] == 1 ? 2 : 3};
          dim3 tpb{8, 8, 4};
          if (dim == 2) {
            tpb = {16, 16, 1};
          }
          const dim3 bpg{(extent[0] - 1) / tpb.x + 1, (extent[1] - 1) / tpb.y + 1, (extent[2] - 1) / tpb.z + 1};
          switch (parameter.get_int("RANS_model")) {
            case 1:
            case 2: //SST
              SST::implicit_treat<<<bpg, tpb>>>(d_ptr);
              break;
            default:break;
          }
        }
      }
      return;
    case 1: // DPLUR
      DPLUR<mixture_model, turb_method>(block, param, d_ptr, h_ptr, parameter);
    default:return;
  }
}
}