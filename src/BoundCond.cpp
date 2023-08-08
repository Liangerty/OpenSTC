#include "BoundCond.h"
#include "Transport.cuh"
#include "gxl_lib/MyString.h"
#include <cmath>

cfd::Inflow::Inflow(const std::string &inflow_name, Species &spec, Parameter &parameter) {
  auto &info = parameter.get_struct(inflow_name);
  label = std::get<integer>(info.at("label"));
  // In default, the mach number, pressure and temperature should be given.
  // If other combinations are given, then implement it later.
  // Currently, 2 combinations are achieved. One is to give (mach, pressure,
  // temperature) The other is to give (density, velocity, pressure)
  if (info.find("mach") != info.end()) mach = std::get<real>(info.at("mach"));
  if (info.find("pressure") != info.end()) pressure = std::get<real>(info.at("pressure"));
  if (info.find("temperature") != info.end()) temperature = std::get<real>(info.at("temperature"));
  if (info.find("velocity") != info.end()) velocity = std::get<real>(info.at("velocity"));
  if (info.find("density") != info.end()) density = std::get<real>(info.at("density"));
  if (info.find("u") != info.end()) u = std::get<real>(info.at("u"));
  if (info.find("v") != info.end()) v = std::get<real>(info.at("v"));
  if (info.find("w") != info.end()) w = std::get<real>(info.at("w"));

  const integer n_scalar = parameter.get_int("n_scalar");
  sv = new real[n_scalar];
  for (int i = 0; i < n_scalar; ++i) {
    sv[i] = 0;
  }
  const integer n_spec{spec.n_spec};

  if (n_spec > 0) {
    // Assign the species mass fraction to the corresponding position.
    // Should be done after knowing the order of species.
    for (auto [name, idx]: spec.spec_list) {
      if (info.find(name) != info.cend()) {
        sv[idx] = std::get<real>(info.at(name));
      }
    }
    mw = 0;
    for (int i = 0; i < n_spec; ++i) mw += sv[i] / spec.mw[i];
    mw = 1 / mw;
  }

  if (temperature < 0) {
    // The temperature is not given, thus the density and pressure are given
    temperature = pressure * mw / (density * R_u);
  }
  if (n_spec > 0) {
    viscosity = compute_viscosity(temperature, mw, sv, spec);
  } else {
    viscosity = Sutherland(temperature);
  }

  real gamma{gamma_air};
  if (n_spec > 0) {
    std::vector<real> cpi(n_spec, 0);
    spec.compute_cp(temperature, cpi.data());
    real cp{0}, cv{0};
    for (size_t i = 0; i < n_spec; ++i) {
      cp += sv[i] * cpi[i];
      cv += sv[i] * (cpi[i] - R_u / spec.mw[i]);
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

  if (parameter.get_int("turbulence_method") == 1) {
    // RANS simulation
    if (parameter.get_int("RANS_model") == 2) {
      // SST
      mut = std::get<real>(info.at("turb_viscosity_ratio")) * viscosity;
      if (info.find("turbulence_intensity") != info.end()) {
        // For SST model, we need k and omega. If SA, we compute this for nothing.
        const real turbulence_intensity = std::get<real>(info.at("turbulence_intensity"));
        sv[n_spec] = 1.5 * velocity * velocity * turbulence_intensity * turbulence_intensity;
        sv[n_spec + 1] = density * sv[n_spec] / mut;
      }
    }
  }

  // This should be re-considered later
  if (inflow_name == "freestream") {
    parameter.update_parameter("rho_inf", density);
    parameter.update_parameter("v_inf", velocity);
    parameter.update_parameter("p_inf", pressure);
    std::vector<real> sv_inf(n_scalar, 0);
    for (int l = 0; l < n_scalar; ++l) {
      sv_inf[l] = sv[l];
    }
    parameter.update_parameter("sv_inf", sv_inf);
  }
}

std::tuple<real, real, real, real, real, real> cfd::Inflow::var_info() const {
  return std::make_tuple(density, u, v, w, pressure, temperature);
}

cfd::Wall::Wall(integer type_label, std::ifstream &bc_file) : label(type_label) {
  std::map<std::string, std::string> opt;
  std::map<std::string, double> par;
  std::string input{}, key{}, name{};
  double val{};
  std::istringstream line(input);
  while (gxl::getline_to_stream(bc_file, input, line, gxl::Case::lower)) {
    line >> key;
    if (key == "double") {
      line >> name >> key >> val;
      par.emplace(std::make_pair(name, val));
    } else if (key == "option") {
      line >> name >> key >> key;
      opt.emplace(std::make_pair(name, key));
    }
    if (key == "label" || key == "end") {
      break;
    }
  }
  if (opt.contains("thermal_type")) {
    thermal_type = opt["thermal_type"] == "isothermal" ? ThermalType::isothermal : ThermalType::adiabatic;
  }
  if (thermal_type == ThermalType::isothermal) {
    if (par.contains("temperature")) {
      temperature = par["temperature"];
    } else {
      printf("Isothermal wall does not specify wall temperature, is set as 300K in default.\n");
    }
  }
}

cfd::Wall::Wall(const std::map<std::string, std::variant<std::string, integer, real>> &info)
    : label(std::get<integer>(info.at("label"))) {
  if (info.contains("thermal_type")) {
    thermal_type = std::get<std::string>(info.at("thermal_type")) == "isothermal" ? ThermalType::isothermal
                                                                                  : ThermalType::adiabatic;
  }
  if (thermal_type == ThermalType::isothermal) {
    if (info.contains("temperature")) {
      temperature = std::get<real>(info.at("temperature"));
    } else {
      printf("Isothermal wall does not specify wall temperature, is set as 300K in default.\n");
    }
  }
}

cfd::Outflow::Outflow(integer type_label) : label(type_label) {}

cfd::Symmetry::Symmetry(integer type_label) : label(type_label) {}
