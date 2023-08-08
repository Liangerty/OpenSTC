#pragma once

#include "Parameter.h"
#include "Define.h"
#include "gxl_lib/Matrix.hpp"

namespace cfd {
struct Species;
struct Reaction;
struct DParameter {
  DParameter()=default;
  explicit DParameter(Parameter &parameter,Species& species, Reaction& reaction);

//  integer myid = 0;   // The process id of this process
//  integer dim = 3;  // The dimension of the simulation problem
//  integer n_block=1;  // number of blocks in this process
  integer inviscid_scheme=0;  // The tag for inviscid scheme. 3 - AUSM+
  integer reconstruction=2; // The reconstruction method for inviscid flux computation
  integer limiter=0;  // The tag for limiter method
  integer viscous_scheme=0; // The tag for viscous scheme. 0 - Inviscid, 2 - 2nd order central discretization
  integer temporal_scheme=0;  // The tag for temporal scheme. 1 - 1st order explicit Euler
//  integer output_screen=10; // determine the interval between screen outputs
  real Pr=0.72;
  real cfl=1;
  integer n_spec=0;
  real* mw = nullptr;
  ggxl::MatrixDyn<real> high_temp_coeff, low_temp_coeff;
  real* t_low = nullptr, * t_mid = nullptr, * t_high = nullptr;
  real* LJ_potent_inv = nullptr;
  real* vis_coeff = nullptr;
  ggxl::MatrixDyn<real> WjDivWi_to_One4th;
  ggxl::MatrixDyn<real> sqrt_WiDivWjPl1Mul8;
  real Sc=0.9;
};
}
