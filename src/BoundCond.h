#pragma once

#include "ChemData.h"
#include "Constants.h"

namespace cfd {

struct Inflow{
  explicit Inflow(const std::string &inflow_name, Species &spec, Parameter &parameter);

  [[nodiscard]] std::tuple<real, real, real, real, real, real> var_info() const;

  void copy_to_gpu(Inflow *d_inflow, Species &spec, const Parameter &parameter);

  integer label = 5;
  real mach = -1;
  real pressure = 101325;
  real temperature = -1;
  real velocity = 0;
  real density = -1;
  real u = 1, v = 0, w = 0;
  real *sv = nullptr;
  real mw = mw_air;
  real viscosity = 0;
  real mut = 0;
};

struct Wall {
  explicit Wall(integer type_label, std::ifstream &bc_file);

  explicit Wall(const std::map<std::string, std::variant<std::string, integer, real>> &info);

  enum class ThermalType { isothermal, adiabatic };

  integer label = 2;
  ThermalType thermal_type = ThermalType::isothermal;
  real temperature{300};
};

struct Outflow {
  explicit Outflow(integer type_label);

  integer label = 6;
};

struct Symmetry {
  explicit Symmetry(integer type_label);

  integer label = 3;
};

}
