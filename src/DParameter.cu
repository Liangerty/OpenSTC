#include "DParameter.h"
#include "ChemData.h"

cfd::DParameter::DParameter(cfd::Parameter &parameter, Species &species, Reaction &reaction) :
    myid{parameter.get_int("myid")}, inviscid_scheme{parameter.get_int("inviscid_scheme")},
    reconstruction{parameter.get_int("reconstruction")}, limiter{parameter.get_int("limiter")},
    viscous_scheme{parameter.get_int("viscous_order")}, rans_model{parameter.get_int("RANS_model")},
    turb_implicit{parameter.get_int("turb_implicit")}, chemSrcMethod{parameter.get_int("chemSrcMethod")},
    Pr(parameter.get_real("prandtl_number")), cfl(parameter.get_real("cfl")),
    Prt(parameter.get_real("turbulent_prandtl_number")), Sct(parameter.get_real("turbulent_schmidt_number")) {
  const auto &spec = species;
  n_spec = spec.n_spec;
  n_scalar = parameter.get_int("n_scalar");
  n_reac = reaction.n_reac;

  // species info
  auto mem_sz = n_spec * sizeof(real);
  cudaMalloc(&mw, mem_sz);
  cudaMemcpy(mw, spec.mw.data(), mem_sz, cudaMemcpyHostToDevice);
  high_temp_coeff.init_with_size(n_spec, 7);
  cudaMemcpy(high_temp_coeff.data(), spec.high_temp_coeff.data(), high_temp_coeff.size() * sizeof(real),
             cudaMemcpyHostToDevice);
  low_temp_coeff.init_with_size(n_spec, 7);
  cudaMemcpy(low_temp_coeff.data(), spec.low_temp_coeff.data(), low_temp_coeff.size() * sizeof(real),
             cudaMemcpyHostToDevice);
  cudaMalloc(&t_low, mem_sz);
  cudaMalloc(&t_mid, mem_sz);
  cudaMalloc(&t_high, mem_sz);
  cudaMemcpy(t_low, spec.t_low.data(), mem_sz, cudaMemcpyHostToDevice);
  cudaMemcpy(t_mid, spec.t_mid.data(), mem_sz, cudaMemcpyHostToDevice);
  cudaMemcpy(t_high, spec.t_high.data(), mem_sz, cudaMemcpyHostToDevice);
  cudaMalloc(&LJ_potent_inv, mem_sz);
  cudaMemcpy(LJ_potent_inv, spec.LJ_potent_inv.data(), mem_sz, cudaMemcpyHostToDevice);
  cudaMalloc(&vis_coeff, mem_sz);
  cudaMemcpy(vis_coeff, spec.vis_coeff.data(), mem_sz, cudaMemcpyHostToDevice);
  WjDivWi_to_One4th.init_with_size(n_spec, n_spec);
  cudaMemcpy(WjDivWi_to_One4th.data(), spec.WjDivWi_to_One4th.data(), WjDivWi_to_One4th.size() * sizeof(real),
             cudaMemcpyHostToDevice);
  sqrt_WiDivWjPl1Mul8.init_with_size(n_spec, n_spec);
  cudaMemcpy(sqrt_WiDivWjPl1Mul8.data(), spec.sqrt_WiDivWjPl1Mul8.data(),
             sqrt_WiDivWjPl1Mul8.size() * sizeof(real), cudaMemcpyHostToDevice);
  Sc = parameter.get_real("schmidt_number");

  // reactions info
  cudaMalloc(&reac_type, n_reac * sizeof(integer));
  cudaMemcpy(reac_type, reaction.label.data(), n_reac * sizeof(integer), cudaMemcpyHostToDevice);
  stoi_f.init_with_size(n_reac, n_spec);
  cudaMemcpy(stoi_f.data(), reaction.stoi_f.data(), stoi_f.size() * sizeof(integer), cudaMemcpyHostToDevice);
  stoi_b.init_with_size(n_reac, n_spec);
  cudaMemcpy(stoi_b.data(), reaction.stoi_b.data(), stoi_b.size() * sizeof(integer), cudaMemcpyHostToDevice);
  mem_sz = n_reac * sizeof(real);
  cudaMalloc(&reac_order, n_reac * sizeof(integer));
  cudaMemcpy(reac_order, reaction.order.data(), n_reac * sizeof(integer), cudaMemcpyHostToDevice);
  cudaMalloc(&A, mem_sz);
  cudaMemcpy(A, reaction.A.data(), mem_sz, cudaMemcpyHostToDevice);
  cudaMalloc(&b, mem_sz);
  cudaMemcpy(b, reaction.b.data(), mem_sz, cudaMemcpyHostToDevice);
  cudaMalloc(&Ea, mem_sz);
  cudaMemcpy(Ea, reaction.Ea.data(), mem_sz, cudaMemcpyHostToDevice);
  cudaMalloc(&A2, mem_sz);
  cudaMemcpy(A2, reaction.A2.data(), mem_sz, cudaMemcpyHostToDevice);
  cudaMalloc(&b2, mem_sz);
  cudaMemcpy(b2, reaction.b2.data(), mem_sz, cudaMemcpyHostToDevice);
  cudaMalloc(&Ea2, mem_sz);
  cudaMemcpy(Ea2, reaction.Ea2.data(), mem_sz, cudaMemcpyHostToDevice);
  third_body_coeff.init_with_size(n_reac, n_spec);
  cudaMemcpy(third_body_coeff.data(), reaction.third_body_coeff.data(), third_body_coeff.size() * sizeof(real),
             cudaMemcpyHostToDevice);
  cudaMalloc(&troe_alpha, mem_sz);
  cudaMemcpy(troe_alpha, reaction.troe_alpha.data(), mem_sz, cudaMemcpyHostToDevice);
  cudaMalloc(&troe_t3, mem_sz);
  cudaMemcpy(troe_t3, reaction.troe_t3.data(), mem_sz, cudaMemcpyHostToDevice);
  cudaMalloc(&troe_t1, mem_sz);
  cudaMemcpy(troe_t1, reaction.troe_t1.data(), mem_sz, cudaMemcpyHostToDevice);
  cudaMalloc(&troe_t2, mem_sz);
  cudaMemcpy(troe_t2, reaction.troe_t2.data(), mem_sz, cudaMemcpyHostToDevice);

  memset(limit_flow.ll, 0, sizeof(real) * LimitFlow::max_n_var);
  memset(limit_flow.ul, 0, sizeof(real) * LimitFlow::max_n_var);
  memset(limit_flow.sv_inf, 0, sizeof(real) * (MAX_SPEC_NUMBER + 2));
  // density limits
  limit_flow.ll[0] = 1e-6 * parameter.get_real("rho_inf");
  limit_flow.ul[0] = 1e+3 * parameter.get_real("rho_inf");
  for (integer l = 1; l < 4; ++l) {
    // u,v,w
    limit_flow.ll[l] = -1e+3 * parameter.get_real("v_inf");
    limit_flow.ul[l] = 1e+3 * parameter.get_real("v_inf");
  }
  // pressure limits
  limit_flow.ll[4] = 1e-6 * parameter.get_real("p_inf");
  limit_flow.ul[4] = 1e+3 * parameter.get_real("p_inf");
  if (rans_model == 2) {
    // SST model
    limit_flow.ul[5] = std::numeric_limits<real>::max();
    limit_flow.ul[6] = std::numeric_limits<real>::max();
  }
  auto &sv_inf{parameter.get_real_array("sv_inf")};
  for (integer l = 0; l < n_scalar; ++l) {
    limit_flow.sv_inf[l] = sv_inf[l];
  }
}
