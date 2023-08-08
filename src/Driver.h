#pragma once

#include <vector>
#include "ChemData.h"
#include "Field.h"
#include "Mesh.h"
#include "Parameter.h"
#include "BoundCond.cuh"
#include "gxl_lib/Time.h"

namespace cfd {
struct DParameter;
class InviscidScheme;
struct ViscousScheme;
struct TemporalScheme;

struct Driver {
  Driver(Parameter &parameter, Mesh &mesh_
#if MULTISPECIES == 1
      , ChemData &chem_data
#endif
  );

  void initialize_computation();

  void simulate();

  integer myid = 0;
  gxl::Time time;
  const Mesh &mesh;
  const Parameter &parameter;
  std::vector<Field> field; // The flowfield data of the simulation. Every block is a "Field" object
#ifdef GPU
  DParameter *param = nullptr; // The parameters used for GPU simulation, datas are stored on GPU while the pointer is on CPU
  DBoundCond bound_cond;  // Boundary conditions
  InviscidScheme **inviscid_scheme = nullptr;
  ViscousScheme **viscous_scheme = nullptr;
  TemporalScheme **temporal_scheme = nullptr;
#endif
  std::array<real, 4> res{1, 1, 1, 1};
  std::array<real, 4> res_scale{1, 1, 1, 1};

private:
  void data_communication();

  void steady_simulation();

  real compute_residual(integer step);

  void steady_screen_output(integer step, real err_max);
};

__global__ void setup_schemes(cfd::InviscidScheme **inviscid_scheme, cfd::ViscousScheme **viscous_scheme,
                              cfd::TemporalScheme **temporal_scheme, cfd::DParameter *param);

template<integer N>
__global__ void reduction_of_dv_squared(real *arr_to_sum, integer size);
}