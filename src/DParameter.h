#pragma once

#include "Parameter.h"
#include "Define.h"
#include "gxl_lib/Matrix.hpp"

namespace cfd {
struct Species;
struct Reaction;

struct DParameter {
  DParameter() = default;

  explicit DParameter(Parameter &parameter, Species &species, Reaction &reaction);

  integer myid = 0;   // The process id of this process
  integer inviscid_scheme = 0;  // The tag for inviscid scheme. 3 - AUSM+
  integer reconstruction = 2; // The reconstruction method for inviscid flux computation
  integer limiter = 0;  // The tag for limiter method
  integer viscous_scheme = 0; // The tag for viscous scheme. 0 - Inviscid, 2 - 2nd order central discretization
  integer rans_model = 0;  // The tag for RANS model. 0 - Laminar, 1 - SA, 2 - SST
  integer turb_implicit = 1;    // If we implicitly treat the turbulent source term. By default, implicitly treat(1), else, 0(explicit)
  integer chemSrcMethod = 0;  // For finite rate chemistry, we need to know how to implicitly treat the chemical source
  integer n_spec = 0;
  integer n_scalar = 0;
  integer n_reac = 0;
  real Pr = 0.72;
  real cfl = 1;
  real *mw = nullptr;
  ggxl::MatrixDyn<real> high_temp_coeff, low_temp_coeff;
  real *t_low = nullptr, *t_mid = nullptr, *t_high = nullptr;
  real *LJ_potent_inv = nullptr;
  real *vis_coeff = nullptr;
  ggxl::MatrixDyn<real> WjDivWi_to_One4th;
  ggxl::MatrixDyn<real> sqrt_WiDivWjPl1Mul8;
  real Sc = 0.9;
  real Prt = 0.9;
  real Sct = 0.9;
  integer *reac_type = nullptr;
  ggxl::MatrixDyn<integer> stoi_f, stoi_b;
  integer *reac_order = nullptr;
  real *A = nullptr, *b = nullptr, *Ea = nullptr;
  real *A2 = nullptr, *b2 = nullptr, *Ea2 = nullptr;
  ggxl::MatrixDyn<real> third_body_coeff;
  real *troe_alpha = nullptr, *troe_t3 = nullptr, *troe_t1 = nullptr, *troe_t2 = nullptr;

private:
  struct LimitFlow {
    // ll for lower limit, ul for upper limit.
    static constexpr integer max_n_var = 5 + 2;
    real ll[max_n_var];
    real ul[max_n_var];
    real sv_inf[MAX_SPEC_NUMBER + 2];
  };

public:
  LimitFlow limit_flow{};
};
}
