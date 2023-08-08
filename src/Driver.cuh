#pragma once

#include "Define.h"
#include "DParameter.h"
#include "gxl_lib/Time.h"
#include "ChemData.h"
#include "Field.h"
#include "BoundCond.cuh"
#include "Output.hpp"

namespace cfd {

template<MixtureModel mix_model, TurbMethod turb_method>
struct Driver {
  Driver(Parameter &parameter, Mesh &mesh_);

  void initialize_computation();

  void simulate();
private:
  void steady_simulation();

  real compute_residual(integer step);

  void steady_screen_output(integer step, real err_max);

public:
  integer myid = 0;
  gxl::Time time;
  const Mesh &mesh;
  const Parameter &parameter;
  Species spec;
  Reaction reac;
  std::vector<cfd::Field<mix_model, turb_method>> field; // The flowfield data of the simulation. Every block is a "Field" object
#ifdef GPU
  DParameter *param = nullptr; // The parameters used for GPU simulation, data are stored on GPU while the pointer is on CPU
  DBoundCond<mix_model, turb_method> bound_cond;  // Boundary conditions
#endif
  std::array<real, 4> res{1, 1, 1, 1};
  std::array<real, 4> res_scale{1, 1, 1, 1};
  Output<mix_model,turb_method> output;
};

template<integer N>
__global__ void reduction_of_dv_squared(real *arr_to_sum, integer size);
} // cfd