#include "ChemData.h"

#include "fmt/core.h"
#include "gxl_lib/MyString.h"
#include "Element.h"
#include "Constants.h"
#include <cmath>

cfd::Species::Species(Parameter &parameter) {
  parameter.update_parameter("n_spec", 0);
  if (parameter.get_bool("species")){
    std::ifstream comb_mech("./input_files/" + parameter.get_string("mechanism_file"));
    std::string input{}, key{};
    gxl::getline(comb_mech, input);  // Elements
    gxl::getline(comb_mech, input);  //--------
    gxl::getline(comb_mech, input);  //$N_e$ elements
    std::istringstream line(input);
    int n_elem{0};
    line >> n_elem;
    gxl::getline(comb_mech, input);  //------
    int counter{0};
    while (gxl::getline(comb_mech, input, gxl::Case::upper)) {
      gxl::to_stringstream(input, line);
      line >> key;
      if (key == "SPECIES") break;
      elem_list.emplace(key, counter++);
      while (line >> key) elem_list.emplace(key, counter++);
    }

    gxl::getline(comb_mech, input);             //------------------
    gxl::getline_to_stream(comb_mech, input, line);  //$N_s$ Species
    int num_spec{0};
    line >> num_spec;
    set_nspec(num_spec, n_elem);
    gxl::getline(comb_mech, input);  //------------------
    parameter.update_parameter("n_spec", num_spec);
    parameter.update_parameter("n_var", parameter.get_int("n_var") + num_spec);
    parameter.update_parameter("n_scalar",parameter.get_int("n_scalar")+num_spec);

    counter = 0;
    while (gxl::getline_to_stream(comb_mech, input, line, gxl::Case::upper)) {
      line >> key;
      if (key == "REACTION") break;
      if (counter >= num_spec) continue;
      register_spec(key, counter);
      while (line >> key && counter < num_spec) register_spec(key, counter);
    }
    comb_mech.close();

    read_therm(parameter);
    read_tran(parameter);

    fmt::print("Mixture composed of {} species will be simulated.\n", n_spec);
    integer counter_spec{0};
    for (auto &[name, label]: spec_list) {
      fmt::print("{}\t", name);
      ++counter_spec;
      if (counter_spec % 10 == 0) {
        fmt::print("\n");
      }
    }
    fmt::print("\n");
  }
}

void cfd::Species::compute_cp(real temp, real *cp) const &{
  const real t2{temp * temp}, t3{t2 * temp}, t4{t3 * temp};
  for (int i = 0; i < n_spec; ++i) {
    real tt{temp};
    if (temp < t_low[i]) {
      tt = t_low[i];
      const real tt2{tt * tt}, tt3{tt2 * tt}, tt4{tt3 * tt};
      auto &coeff = low_temp_coeff;
      cp[i] = coeff(i, 0) + coeff(i, 1) * tt + coeff(i, 2) * tt2 +
              coeff(i, 3) * tt3 + coeff(i, 4) * tt4;
    } else {
      auto &coeff = tt < t_mid[i] ? low_temp_coeff : high_temp_coeff;
      cp[i] = coeff(i, 0) + coeff(i, 1) * tt + coeff(i, 2) * t2 +
              coeff(i, 3) * t3 + coeff(i, 4) * t4;
    }
    cp[i] *= R_u / mw[i];
  }
}

void cfd::Species::set_nspec(integer n_sp, integer n_elem) {
  n_spec = n_sp;
  elem_comp.resize(n_sp, n_elem);
  mw.resize(n_sp, 0);
  t_low.resize(n_sp, 300);
  t_mid.resize(n_sp, 1000);
  t_high.resize(n_sp, 5000);
  high_temp_coeff.resize(n_sp, 7);
  low_temp_coeff.resize(n_sp, 7);
  LJ_potent_inv.resize(n_sp, 0);
  vis_coeff.resize(n_sp, 0);
  WjDivWi_to_One4th.resize(n_sp, n_sp);
  sqrt_WiDivWjPl1Mul8.resize(n_sp, n_sp);
  x.resize(n_sp, 0);
  vis_spec.resize(n_sp, 0);
  lambda.resize(n_sp, 0);
  partition_fun.resize(n_sp, n_sp);
}

void cfd::Species::register_spec(const std::string &name, integer &index) {
  spec_list.emplace(name, index);
  ++index;
}

void cfd::Species::read_therm(Parameter &parameter) {
  std::ifstream therm_dat("./input_files/" + parameter.get_string("therm_file"));
  std::string input{};
  gxl::read_until(therm_dat, input, "THERMO", gxl::Case::upper);  // "THERMO"
  std::getline(therm_dat, input);  //$T_low$ $T_mid$ $T_high$
  std::istringstream line(input);
  double T_low{300}, T_mid{1000}, T_high{5000};
  line >> T_low >> T_mid >> T_high;
  t_low.resize(n_spec, T_low);
  t_mid.resize(n_spec, T_mid);
  t_high.resize(n_spec, T_high);

  std::string key{};
  gxl::getline_to_stream(therm_dat, input, line, gxl::Case::upper);
  line >> key;
  int n_read{0};
  while (key != "END" && n_read < n_spec) {
    if (key == "!" || input[0] == '!' || input.empty()) {
      gxl::getline_to_stream(therm_dat, input, line, gxl::Case::upper);
      line >> key;
      continue;
    }
    key.assign(input, 0, 18);
    gxl::to_stringstream(key, line);
    line >> key;
    if (!spec_list.contains(key)) {
      gxl::getline(therm_dat, input);
      gxl::getline(therm_dat, input);
      gxl::getline(therm_dat, input);
      gxl::getline_to_stream(therm_dat, input, line, gxl::Case::upper);
      line >> key;
      continue;
    }
    const int curr_sp = spec_list.at(key);

    key.assign(input, 45, 10);  // T_low
    t_low[curr_sp] = std::stod(key);
    key.assign(input, 55, 10);  // T_high
    t_high[curr_sp] = std::stod(key);
    key.assign(input, 65, 10);  // Probably specify a different T_mid
    gxl::to_stringstream(key, line);
    line >> key;
    if (!key.empty()) t_mid[curr_sp] = std::stod(key);

    // Read element composition
    std::string comp_str{};
    for (int i = 0; i < 4; ++i) {
      comp_str.assign(input, 24 + i * 5, 5);
      gxl::trim_left(comp_str);
      if (comp_str.empty() || comp_str.starts_with('0')) break;
      gxl::to_stringstream(comp_str, line);
      line >> key;
      int stoi{0};
      line >> stoi;
      elem_comp(curr_sp, elem_list[key]) = stoi;
    }
    // Compute the relative molecular weight
    double mole_weight{0};
    for (const auto &[element, label]: elem_list) {
      mole_weight += Element{element}.get_atom_weight() *
                     elem_comp(curr_sp, label);
    }
    mw[curr_sp] = mole_weight;

    // Read the thermodynamic fitting coefficients
    std::getline(therm_dat, input);
    std::string cs1{}, cs2{}, cs3{}, cs4{}, cs5{};
    double c1, c2, c3, c4, c5;
    cs1.assign(input, 0, 15);
    cs2.assign(input, 15, 15);
    cs3.assign(input, 30, 15);
    cs4.assign(input, 45, 15);
    cs5.assign(input, 60, 15);
    c1 = std::stod(cs1);
    c2 = std::stod(cs2);
    c3 = std::stod(cs3);
    c4 = std::stod(cs4);
    c5 = std::stod(cs5);
    high_temp_coeff(curr_sp, 0) = c1;
    high_temp_coeff(curr_sp, 1) = c2;
    high_temp_coeff(curr_sp, 2) = c3;
    high_temp_coeff(curr_sp, 3) = c4;
    high_temp_coeff(curr_sp, 4) = c5;
    // second line
    std::getline(therm_dat, input);
    cs1.assign(input, 0, 15);
    cs2.assign(input, 15, 15);
    cs3.assign(input, 30, 15);
    cs4.assign(input, 45, 15);
    cs5.assign(input, 60, 15);
    c1 = std::stod(cs1);
    c2 = std::stod(cs2);
    c3 = std::stod(cs3);
    c4 = std::stod(cs4);
    c5 = std::stod(cs5);
    high_temp_coeff(curr_sp, 5) = c1;
    high_temp_coeff(curr_sp, 6) = c2;
    low_temp_coeff(curr_sp, 0) = c3;
    low_temp_coeff(curr_sp, 1) = c4;
    low_temp_coeff(curr_sp, 2) = c5;
    // third line
    std::getline(therm_dat, input);
    cs1.assign(input, 0, 15);
    cs2.assign(input, 15, 15);
    cs3.assign(input, 30, 15);
    cs4.assign(input, 45, 15);
    c1 = std::stod(cs1);
    c2 = std::stod(cs2);
    c3 = std::stod(cs3);
    c4 = std::stod(cs4);
    low_temp_coeff(curr_sp, 3) = c1;
    low_temp_coeff(curr_sp, 4) = c2;
    low_temp_coeff(curr_sp, 5) = c3;
    low_temp_coeff(curr_sp, 6) = c4;

    gxl::getline_to_stream(therm_dat, input, line);
    line >> key;

    ++n_read;
  }
  // fmt::print("Thermodynamic properties are read successfully for {} species.\n", n_spec);
  therm_dat.close();
}

void cfd::Species::read_tran(Parameter &parameter) {
  std::ifstream tran_dat("input_files/" +
                         parameter.get_string("transport_file"));
  std::string input{}, key{};
  gxl::getline(tran_dat, input, gxl::Case::upper);
  std::istringstream line(input);
  line >> key;
  auto &list = spec_list;
  while (!key.starts_with("END")) {
    if (!list.contains(key)) {
      gxl::getline_to_stream(tran_dat, input, line, gxl::Case::upper);
      line >> key;
      continue;
    }
    gxl::to_stringstream(input, line);
    double lj_potential{0}, collision_diameter{0}, pass{0};
    line >> key >> pass >> lj_potential >> collision_diameter;
    const int idx = list.at(key);
    LJ_potent_inv[idx] = 1.0 / lj_potential;
    vis_coeff[idx] =
        2.6693e-6 * sqrt(mw[idx]) / (collision_diameter * collision_diameter);
    gxl::getline_to_stream(tran_dat, input, line, gxl::Case::upper);
    line >> key;
  }

  for (int i = 0; i < n_spec; ++i) {
    for (int j = 0; j < n_spec; ++j) {
      WjDivWi_to_One4th(i, j) = std::pow(mw[j] / mw[i], 0.25);
      sqrt_WiDivWjPl1Mul8(i, j) = 1.0 / std::sqrt(8 * (1 + mw[i] / mw[j]));
    }
  }
}

cfd::Reaction::Reaction(Parameter &parameter) {}

cfd::ChemData::ChemData(Parameter &parameter) : spec(parameter), reac(parameter) {}

