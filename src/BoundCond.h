#pragma once

#include "Constants.h"
#include "Define.h"
#include "Parameter.h"
#include "ChemData.h"
#include "Mesh.h"
#include "Transport.cuh"

namespace cfd {
template<MixtureModel mix_model=MixtureModel::Air, TurbMethod turb_method=TurbMethod::Laminar>
struct Inflow{
  explicit Inflow(const std::map<std::string, std::variant<std::string, integer, real>> &info,
                  Species &spec);

  [[nodiscard]] std::tuple<real, real, real, real, real, real> var_info() const;

  void copy_to_gpu(Inflow *d_inflow, Species &spec);

  integer label = 5;
  real mach = -1;
  real pressure = 101325;
  real temperature = -1;
  real velocity = 0;
  real density = -1;
  real u = 1, v = 0, w = 0;
  real *yk = nullptr;
  real mw = mw_air;
  real viscosity = 0;
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

// Below are implementations of above functions
template<MixtureModel mix_model, TurbMethod turb_method>
cfd::Inflow<mix_model, turb_method>::Inflow(
    const std::map<std::string, std::variant<std::string, integer, real>> &info, Species &spec):label{
    std::get<integer>(info.at("label"))} {
  // In default, the mach number, pressure and temperature should be given.
  // If other combinations are given, then implement it later.
  // Currently, 2 combinations are achieved. One is to give (mach, pressure,
  // temperature) The other is to give (density, velocity, pressure)
  if (info.find("mach")!=info.end()) mach = std::get<real>(info.at("mach"));
  if (info.find("pressure")!=info.end()) pressure = std::get<real>(info.at("pressure"));
  if (info.find("temperature")!=info.end()) temperature = std::get<real>(info.at("temperature"));
  if (info.find("velocity")!=info.end()) velocity = std::get<real>(info.at("velocity"));
  if (info.find("density")!=info.end()) density = std::get<real>(info.at("density"));
  if (info.find("u")!=info.end()) u = std::get<real>(info.at("u"));
  if (info.find("v")!=info.end()) v = std::get<real>(info.at("v"));
  if (info.find("w")!=info.end()) w = std::get<real>(info.at("w"));

  if constexpr (mix_model==MixtureModel::Mixture){
    const integer n_spec{spec.n_spec};
    yk = new real[n_spec];
    for (int qq = 0; qq < n_spec; ++qq) {
      yk[qq] = 0;
    }

    // Assign the species mass fraction to the corresponding position.
    // Should be done after knowing the order of species.
    for (auto [name, idx]: spec.spec_list) {
      if (info.find(name)!=info.cend()) {
        yk[idx] = std::get<real>(info.at(name));
      }
    }
    mw = 0;
    for (int i = 0; i < n_spec; ++i) mw += yk[i] / spec.mw[i];
    mw = 1 / mw;
  }

  if (temperature < 0) {
    // The temperature is not given, thus the density and pressure are given
    temperature = pressure * mw / (density * R_u);
  }
  if constexpr (mix_model==MixtureModel::Mixture){
    viscosity = compute_viscosity(temperature, mw, yk, spec);
  }else{
    viscosity = Sutherland(temperature);
  }

  real gamma{gamma_air};
  if constexpr (mix_model==MixtureModel::Mixture){
    const integer n_spec{spec.n_spec};
    std::vector<real> cpi(n_spec, 0);
    spec.compute_cp(temperature, cpi.data());
    real cp{0}, cv{0};
    for (size_t i = 0; i < n_spec; ++i) {
      cp += yk[i] * cpi[i];
      cv += yk[i] * (cpi[i] - R_u / spec.mw[i]);
    }
    gamma = cp / cv;  // specific heat ratio
  }

  const real c{std::sqrt(gamma * R_u / mw * temperature)};  // speed of sound

  if (mach < 0) {
    // The mach number is not given. The velocity magnitude should be given
    mach = velocity / c;
  } else {
    // The velocity magnitude is not given. The mach number should be given
    velocity = mach * c;
  }
  u *= velocity;
  v *= velocity;
  w *= velocity;
  if (density < 0) {
    // The density is not given, compute it from equation of state
    density = pressure * mw / (R_u * temperature);
  }
}

template<MixtureModel mix_model, TurbMethod turb_method>
std::tuple<real, real, real, real, real, real> cfd::Inflow<mix_model, turb_method>::var_info() const{
  return std::make_tuple(density, u, v, w, pressure, temperature);
}
}
