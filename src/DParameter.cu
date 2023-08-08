#include "DParameter.h"
#include "ChemData.h"

cfd::DParameter::DParameter(cfd::Parameter &parameter,Species& species, Reaction& reaction) : /*myid{parameter.get_int("myid")},
                                                                              dim{parameter.get_int("dimension")},
                                                                              n_block(parameter.get_int("n_block")),*/
                                                                              inviscid_scheme{
                                                                                  parameter.get_int("inviscid_scheme")},
                                                                              reconstruction{
                                                                                  parameter.get_int("reconstruction")},
                                                                              limiter{parameter.get_int("limiter")},
                                                                              viscous_scheme{
                                                                                  parameter.get_int("viscous_order")},
                                                                              temporal_scheme{
                                                                                  parameter.get_int("temporal_scheme")},
                                                                              /*output_screen(
                                                                                  parameter.get_int("output_screen")),*/
                                                                              Pr(parameter.get_real("prandtl_number")),
                                                                              cfl(parameter.get_real("cfl")) {
  const auto &spec = species;
  n_spec = spec.n_spec;
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
}
