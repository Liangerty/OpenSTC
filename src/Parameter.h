#pragma once

#include <unordered_map>
#include <vector>
#include <string>
#include <array>
#include "Define.h"
#include <map>
#include <variant>

namespace cfd {

class MpiParallel;

class Parameter {
  std::unordered_map<std::string, integer> int_parameters{};
  std::unordered_map<std::string, real> real_parameters{};
  std::unordered_map<std::string, bool> bool_parameters{};
  std::unordered_map<std::string, std::string> string_parameters{};
  std::unordered_map<std::string, std::vector<int>> int_array{};
  std::unordered_map<std::string, std::vector<real>> real_array{};
  std::unordered_map<std::string, std::vector<std::string>> string_array{};
  std::unordered_map<std::string, std::map<std::string, std::variant<std::string, integer, real>>> struct_array;
public:
  explicit Parameter(const MpiParallel &mpi_parallel);

  explicit Parameter(const std::string &filename);

  Parameter(const Parameter &) = delete;

  Parameter(Parameter &&) = delete;

  Parameter operator=(Parameter &&) = delete;

  Parameter &operator=(const Parameter &) = delete;

  int &get_int(const std::string &name) { return int_parameters.at(name); }

  [[nodiscard]] const int &get_int(const std::string &name) const { return int_parameters.at(name); }

  real &get_real(const std::string &name) { return real_parameters.at(name); }

  [[nodiscard]] const real &get_real(const std::string &name) const { return real_parameters.at(name); }

  bool &get_bool(const std::string &name) { return bool_parameters.at(name); }

  [[nodiscard]] const bool &get_bool(const std::string &name) const { return bool_parameters.at(name); }

  std::string &get_string(const std::string &name) { return string_parameters.at(name); }

  [[nodiscard]] const auto& get_struct(const std::string &name)const{return struct_array.at(name);}

  [[nodiscard]] const auto& get_string_array(const std::string &name)const{return string_array.at(name);}
  [[nodiscard]] const auto& get_real_array(const std::string &name)const{return real_array.at(name);}
  [[nodiscard]] const auto& get_int_array(const std::string &name)const{return int_array.at(name);}

  void update_parameter(const std::string &name, const int new_value) { int_parameters[name] = new_value; }
  void update_parameter(const std::string &name, const real new_value) { real_parameters[name] = new_value; }
  void update_parameter(const std::string &name, const std::vector<real>& new_value) { real_array[name] = new_value; }

  ~Parameter() = default;

private:
  const std::array<std::string, 9> file_names{
      "./input_files/setup/0_global_control.txt",   //basic information about the simulation
      "./input_files/setup/1_grid_information.txt", //the information about grid
      "./input_files/setup/2_scheme.txt",
      "./input_files/setup/3_species_reactions.txt",
      "./input_files/setup/4_laminar_turbulent.txt",
      "./input_files/setup/5_boundary_condition.txt",
      "./input_files/setup/6_post_process.txt",
      "./input_files/setup/8_initialization.txt",
      "./input_files/setup/9_transport_property.txt"
  };

  void read_param_from_file();

  void read_one_file(std::ifstream &file);

  template<typename T>
  integer read_line_to_array(std::istringstream& line, std::vector<T>& arr);

  static std::map<std::string, std::variant<std::string, integer, real>> read_struct(std::ifstream &file);
};
}
