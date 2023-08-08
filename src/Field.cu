#include "Field.h"
#include "BoundCond.h"

cfd::Field::Field(Parameter &parameter, const Block &block_in) : block(block_in) {
  const integer mx{block.mx}, my{block.my}, mz{block.mz}, ngg{block.ngg};
  n_var = parameter.get_int("n_var");
  integer n_scalar{0};
  integer n_other_var{1}; // Default, mach number

  bv.resize(mx, my, mz, 6, ngg);
  n_scalar += parameter.get_int("n_spec");
  if (parameter.get_int("turbulence_method") == 1) {
    // RANS
    n_scalar += parameter.get_int("n_turb");
    n_other_var += 1; // mut
  }
  sv.resize(mx, my, mz, n_scalar, ngg);
  ov.resize(mx, my, mz, n_other_var, ngg);
}

void cfd::Field::initialize_basic_variables(const Parameter &parameter, const std::vector<Inflow> &inflows,
                                            const std::vector<real> &xs, const std::vector<real> &xe,
                                            const std::vector<real> &ys, const std::vector<real> &ye,
                                            const std::vector<real> &zs, const std::vector<real> &ze) {
  const auto n = inflows.size();
  std::vector<real> rho(n, 0), u(n, 0), v(n, 0), w(n, 0), p(n, 0), T(n, 0);
  const integer n_scalar = parameter.get_int("n_scalar");
  gxl::MatrixDyn<real> scalar_inflow{static_cast<int>(n), n_scalar};

  for (size_t i = 0; i < inflows.size(); ++i) {
    std::tie(rho[i], u[i], v[i], w[i], p[i], T[i]) = inflows[i].var_info();
  }
  for (size_t i = 0; i < inflows.size(); ++i) {
    auto sv_this = inflows[i].sv;
    for (int l = 0; l < n_scalar; ++l) {
      scalar_inflow(static_cast<int>(i), l) = sv_this[l];
    }
  }

  const int ngg{block.ngg};
  for (int i = -ngg; i < block.mx + ngg; ++i) {
    for (int j = -ngg; j < block.my + ngg; ++j) {
      for (int k = -ngg; k < block.mz + ngg; ++k) {
        size_t i_init{0};
        if (inflows.size() > 1) {
          for (size_t l = 1; l < inflows.size(); ++l) {
            if (block.x(i, j, k) >= xs[l] && block.x(i, j, k) <= xe[l]
                && block.y(i, j, k) >= ys[l] && block.y(i, j, k) <= ye[l]
                && block.z(i, j, k) >= zs[l] && block.z(i, j, k) <= ze[l]) {
              i_init = l;
              break;
            }
          }
        }
        bv(i, j, k, 0) = rho[i_init];
        bv(i, j, k, 1) = u[i_init];
        bv(i, j, k, 2) = v[i_init];
        bv(i, j, k, 3) = w[i_init];
        bv(i, j, k, 4) = p[i_init];
        bv(i, j, k, 5) = T[i_init];
        for (integer l = 0; l < n_scalar; ++l) {
          sv(i, j, k, l) = scalar_inflow(static_cast<int>(i_init), l);
        }
      }
    }
  }
}

void cfd::Field::setup_device_memory(const Parameter &parameter) {
  h_ptr = new DZone;
  h_ptr->mx = block.mx, h_ptr->my = block.my, h_ptr->mz = block.mz, h_ptr->ngg = block.ngg;

  h_ptr->x.allocate_memory(h_ptr->mx, h_ptr->my, h_ptr->mz, h_ptr->ngg);
  cudaMemcpy(h_ptr->x.data(), block.x.data(), sizeof(real) * h_ptr->x.size(), cudaMemcpyHostToDevice);
  h_ptr->y.allocate_memory(h_ptr->mx, h_ptr->my, h_ptr->mz, h_ptr->ngg);
  cudaMemcpy(h_ptr->y.data(), block.y.data(), sizeof(real) * h_ptr->y.size(), cudaMemcpyHostToDevice);
  h_ptr->z.allocate_memory(h_ptr->mx, h_ptr->my, h_ptr->mz, h_ptr->ngg);
  cudaMemcpy(h_ptr->z.data(), block.z.data(), sizeof(real) * h_ptr->z.size(), cudaMemcpyHostToDevice);

  auto n_bound{block.boundary.size()};
  auto n_inner{block.inner_face.size()};
  auto n_par{block.parallel_face.size()};
  auto mem_sz = sizeof(Boundary) * n_bound;
  cudaMalloc(&h_ptr->boundary, mem_sz);
  cudaMemcpy(h_ptr->boundary, block.boundary.data(), mem_sz, cudaMemcpyHostToDevice);
  mem_sz = sizeof(InnerFace) * n_inner;
  cudaMalloc(&h_ptr->innerface, mem_sz);
  cudaMemcpy(h_ptr->innerface, block.inner_face.data(), mem_sz, cudaMemcpyHostToDevice);
  mem_sz = sizeof(ParallelFace) * n_par;
  cudaMalloc(&h_ptr->parface, mem_sz);
  cudaMemcpy(h_ptr->parface, block.parallel_face.data(), mem_sz, cudaMemcpyHostToDevice);

  h_ptr->jac.allocate_memory(h_ptr->mx, h_ptr->my, h_ptr->mz, h_ptr->ngg);
  cudaMemcpy(h_ptr->jac.data(), block.jacobian.data(), sizeof(real) * h_ptr->jac.size(), cudaMemcpyHostToDevice);
  h_ptr->metric.allocate_memory(h_ptr->mx, h_ptr->my, h_ptr->mz, h_ptr->ngg);
  cudaMemcpy(h_ptr->metric.data(), block.metric.data(), sizeof(gxl::Matrix<real, 3, 3, 1>) * h_ptr->metric.size(),
             cudaMemcpyHostToDevice);

  h_ptr->n_var = parameter.get_int("n_var");
  h_ptr->cv.allocate_memory(h_ptr->mx, h_ptr->my, h_ptr->mz, h_ptr->n_var, h_ptr->ngg);
  h_ptr->bv.allocate_memory(h_ptr->mx, h_ptr->my, h_ptr->mz, 6, h_ptr->ngg);
  cudaMemcpy(h_ptr->bv.data(), bv.data(), sizeof(real) * h_ptr->bv.size() * 6, cudaMemcpyHostToDevice);
  h_ptr->bv_last.allocate_memory(h_ptr->mx, h_ptr->my, h_ptr->mz, 4, 0);
  h_ptr->vel.allocate_memory(h_ptr->mx, h_ptr->my, h_ptr->mz, h_ptr->ngg);
  h_ptr->acoustic_speed.allocate_memory(h_ptr->mx, h_ptr->my, h_ptr->mz, h_ptr->ngg);
  h_ptr->mach.allocate_memory(h_ptr->mx, h_ptr->my, h_ptr->mz, h_ptr->ngg);
  h_ptr->mul.allocate_memory(h_ptr->mx, h_ptr->my, h_ptr->mz, h_ptr->ngg);
  h_ptr->thermal_conductivity.allocate_memory(h_ptr->mx, h_ptr->my, h_ptr->mz, h_ptr->ngg);

  h_ptr->n_spec = parameter.get_int("n_spec");
  h_ptr->n_scal = parameter.get_int("n_scalar");
  h_ptr->sv.allocate_memory(h_ptr->mx, h_ptr->my, h_ptr->mz, h_ptr->n_scal, h_ptr->ngg);
  cudaMemcpy(h_ptr->sv.data(), sv.data(), sizeof(real) * h_ptr->sv.size() * h_ptr->n_scal, cudaMemcpyHostToDevice);
  h_ptr->rho_D.allocate_memory(h_ptr->mx, h_ptr->my, h_ptr->mz, h_ptr->n_spec, h_ptr->ngg);
  if (h_ptr->n_spec > 0) {
    h_ptr->gamma.allocate_memory(h_ptr->mx, h_ptr->my, h_ptr->mz, h_ptr->ngg);
    h_ptr->cp.allocate_memory(h_ptr->mx, h_ptr->my, h_ptr->mz, h_ptr->ngg);
    if (parameter.get_int("n_reac") > 0) {
      // Finite rate chemistry
      if (const integer chemSrcMethod = parameter.get_int("chemSrcMethod");chemSrcMethod == 1) {
        // EPI
        h_ptr->chem_src_jac.allocate_memory(h_ptr->mx, h_ptr->my, h_ptr->mz, h_ptr->n_spec * h_ptr->n_spec, 0);
      } else if (chemSrcMethod == 2) {
        // DA
        h_ptr->chem_src_jac.allocate_memory(h_ptr->mx, h_ptr->my, h_ptr->mz, h_ptr->n_spec, 0);
      }
    }
  }
  if (parameter.get_int("turbulence_method") == 1) {
    // RANS method
    h_ptr->mut.allocate_memory(h_ptr->mx, h_ptr->my, h_ptr->mz, h_ptr->ngg);
    h_ptr->turb_therm_cond.allocate_memory(h_ptr->mx, h_ptr->my, h_ptr->mz, h_ptr->ngg);
    if (parameter.get_int("RANS_model") == 2) {
      // SST
      h_ptr->wall_distance.allocate_memory(h_ptr->mx, h_ptr->my, h_ptr->mz, h_ptr->ngg);
      if (parameter.get_int("turb_implicit") == 1) {
        h_ptr->turb_src_jac.allocate_memory(h_ptr->mx, h_ptr->my, h_ptr->mz, 2, 0);
      }
    }
  }
//  if constexpr (turb_method == TurbMethod::RANS) {
//    h_ptr->mut.allocate_memory(h_ptr->mx, h_ptr->my, h_ptr->mz, h_ptr->ngg);
//    h_ptr->turb_therm_cond.allocate_memory(h_ptr->mx, h_ptr->my, h_ptr->mz, h_ptr->ngg);
//    if (parameter.get_int("RANS_model") == 2) {
//      // SST
//      h_ptr->wall_distance.allocate_memory(h_ptr->mx, h_ptr->my, h_ptr->mz, h_ptr->ngg);
//      if (parameter.get_int("turb_implicit") == 1) {
//        h_ptr->turb_src_jac.allocate_memory(h_ptr->mx, h_ptr->my, h_ptr->mz, 2, 0);
//      }
//    }
//  }

  h_ptr->dq.allocate_memory(h_ptr->mx, h_ptr->my, h_ptr->mz, h_ptr->n_var, 0);
  h_ptr->inv_spectr_rad.allocate_memory(h_ptr->mx, h_ptr->my, h_ptr->mz, 0);
  h_ptr->visc_spectr_rad.allocate_memory(h_ptr->mx, h_ptr->my, h_ptr->mz, 0);
  if (parameter.get_int("implicit_method") == 1) {//DPLUR
    // If DPLUR type, when computing the products of convective jacobian and dq, we need 1 layer of ghost grids whose dq=0.
    // Except those inner or parallel comnnunication faces, they need to get the dq from neighbor blocks.
    h_ptr->dq.allocate_memory(h_ptr->mx, h_ptr->my, h_ptr->mz, h_ptr->n_var, 1);
    h_ptr->dq0.allocate_memory(h_ptr->mx, h_ptr->my, h_ptr->mz, h_ptr->n_var, 1);
    h_ptr->dqk.allocate_memory(h_ptr->mx, h_ptr->my, h_ptr->mz, h_ptr->n_var, 1);
    h_ptr->inv_spectr_rad.allocate_memory(h_ptr->mx, h_ptr->my, h_ptr->mz, 1);
    h_ptr->visc_spectr_rad.allocate_memory(h_ptr->mx, h_ptr->my, h_ptr->mz, 1);
  }
  if (parameter.get_bool("steady")) { // steady simulation
    h_ptr->dt_local.allocate_memory(h_ptr->mx, h_ptr->my, h_ptr->mz, 0);
  }

  cudaMalloc(&d_ptr, sizeof(DZone));
  cudaMemcpy(d_ptr, h_ptr, sizeof(DZone), cudaMemcpyHostToDevice);
}

void cfd::Field::copy_data_from_device(const Parameter &parameter) {
  const auto size = (block.mx + 2 * block.ngg) * (block.my + 2 * block.ngg) * (block.mz + 2 * block.ngg);

  cudaMemcpy(bv.data(), h_ptr->bv.data(), 6 * size * sizeof(real), cudaMemcpyDeviceToHost);
  cudaMemcpy(ov.data(), h_ptr->mach.data(), size * sizeof(real), cudaMemcpyDeviceToHost);
  if (parameter.get_int("turbulence_method") == 1) {
    cudaMemcpy(ov[1], h_ptr->mut.data(), size * sizeof(real), cudaMemcpyDeviceToHost);
  }
  cudaMemcpy(sv.data(), h_ptr->sv.data(), h_ptr->n_scal * size * sizeof(real), cudaMemcpyDeviceToHost);
}
