#pragma once

#include "Define.h"
#include "Parameter.h"
#include "gxl_lib/Matrix.hpp"

namespace cfd {
struct Species {
  explicit Species(Parameter &parameter);

  integer n_spec{0};  // number of species
  std::map<std::string, integer> spec_list; // species list

  void compute_cp(real temp, real *cp) const &;

  // The properties of the species. Some previously private derived variables will appear in the corresponding function classes.
  std::map<std::string, integer> elem_list; // element list
  gxl::MatrixDyn<integer> elem_comp;  // the element composition of the species
  std::vector<real> mw; // the array of molecular weights
  // Thermodynamic properties
  std::vector<real> t_low, t_mid, t_high; // the array of thermodynamic sections
  gxl::MatrixDyn<real> high_temp_coeff, low_temp_coeff; // the cp/h/s polynomial coefficients
  // Transport properties
  std::vector<real> LJ_potent_inv;  // the inverse of the Lennard-Jones potential
  std::vector<real> vis_coeff;  // the coefficient to compute viscosity
  gxl::MatrixDyn<real> WjDivWi_to_One4th, sqrt_WiDivWjPl1Mul8; // Some constant value to compute partition functions
  // Temporary variables to compute transport properties, should not be accessed from other functions
  std::vector<real> x;
  std::vector<real> vis_spec;
  std::vector<real> lambda;
  gxl::MatrixDyn<real> partition_fun;


private:
  void set_nspec(integer n_sp, integer n_elem);

  bool read_therm(std::ifstream &therm_dat, bool read_from_comb_mech);

  void read_tran(std::ifstream &tran_dat);
};

struct Reaction {
  explicit Reaction(Parameter &parameter, const Species& species);

private:
  void set_nreac(integer nr, integer ns);

  void read_reaction_line(std::string input, integer idx, const Species& species);

  std::string get_auxi_info(std::ifstream &file, integer idx, const cfd::Species &species, bool &is_dup);

public:
  integer n_reac{0};
  // The label represents which method to compute kf and kb.
  // 0 - Irreversible, 1 - Reversible
  // 2 - REV (reversible with both kf and kb Arrhenius coefficients given)
  // 3 - DUP (Multiple sets of kf Arrhenius coefficients given)
  // 4 - Third body reactions ( +M is added on both sides, indicating the reaction needs catylists)
  // 5 - Lindemann Type (Pressure dependent reactions computed with Lindemann type method)
  // 6 - Troe-3 (Pressure dependent reactions computed with Troe type method, 3 parameters)
  // 7 - Troe-4 (Pressure dependent reactions computed with Troe type method, 4 parameters)
  std::vector<integer> label;
  gxl::MatrixDyn<integer> stoi_f, stoi_b;
  std::vector<integer> order;
  std::vector<real> A, b, Ea;
  std::vector<real> A2, b2, Ea2;
  gxl::MatrixDyn<real> third_body_coeff;
  std::vector<real> troe_alpha, troe_t3, troe_t1, troe_t2;
};
}
