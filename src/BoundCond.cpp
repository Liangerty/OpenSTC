#include "BoundCond.h"
#include "gxl_lib/MyString.h"

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
