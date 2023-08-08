#pragma once

#include "Parameter.h"
#include "Field.h"
#include "ChemData.h"
#include <fstream>
#include <filesystem>
#include "gxl_lib/MyString.h"


namespace cfd {
class Mesh;

template<MixtureModel mix_model = MixtureModel::Air, TurbMethod turb_method = TurbMethod::Laminar>
void
initialize_basic_variables(Parameter &parameter, const Mesh &mesh, std::vector<Field<mix_model, turb_method>> &field,
                           Species &species);

template<MixtureModel mix_model = MixtureModel::Air, TurbMethod turb_method = TurbMethod::Laminar>
void initialize_from_start(Parameter &parameter, const Mesh &mesh, std::vector<Field<mix_model, turb_method>> &field,
                           Species &species);

template<MixtureModel mix_model = MixtureModel::Air, TurbMethod turb_method = TurbMethod::Laminar>
void read_flowfield(Parameter &parameter, const Mesh &mesh, std::vector<Field<mix_model, turb_method>> &field,
                    Species &species);

/**
 * @brief To relate the order of variables from the flowfield files to bv, yk, turbulent arrays
 * @param var_name the array which contains all variables from the flowfield files
 * @return an array of orders. 0~5 means density, u, v, w, p, T; 6~5+ns means the species order, 6+ns~... means other variables such as mut...
 */
template<MixtureModel mix_model, TurbMethod turb_method>
std::vector<integer>
identify_variable_labels(std::vector<std::string> &var_name, Species &species, Parameter &parameter,
                         std::array<integer, 2> &old_data_info);

void read_one_useless_variable(FILE *fp, integer mx, integer my, integer mz, integer data_format);

template<MixtureModel mix_model, TurbMethod turb_method>
void initialize_spec_from_inflow(cfd::Parameter &parameter, const cfd::Mesh &mesh,
                                 std::vector<Field<mix_model, turb_method>> &field, Species &species);

// Implementations
template<MixtureModel mix_model, TurbMethod turb_method>
void
initialize_basic_variables(Parameter &parameter, const Mesh &mesh, std::vector<Field<mix_model, turb_method>> &field,
                           Species &species) {
  const integer init_method = parameter.get_int("initial");
  switch (init_method) {
    case 0:initialize_from_start(parameter, mesh, field, species);
      break;
    case 1:read_flowfield(parameter, mesh, field, species);
      break;
    default:printf("The initialization method is unknown, use freestream value to initialize by default.\n");
      initialize_from_start(parameter, mesh, field, species);
  }
}

template<MixtureModel mix_model, TurbMethod turb_method>
void
initialize_from_start(Parameter &parameter, const Mesh &mesh, std::vector<Field<mix_model, turb_method>> &field,
                      Species &species) {
  // First, find out how many groups of initial conditions are needed.
  const integer tot_group{parameter.get_int("groups_init")};
  std::vector<Inflow<mix_model, turb_method>> groups_inflow;
  const std::string default_init = parameter.get_string("default_init");
  Inflow<mix_model, turb_method> default_inflow(parameter.get_struct(default_init), species);
  groups_inflow.push_back(default_inflow);

  std::vector<real> xs{}, xe{}, ys{}, ye{}, zs{}, ze{};
  if (tot_group > 1) {
    for (integer l = 0; l < tot_group - 1; ++l) {
      auto patch_struct_name = "init_cond_" + std::to_string(l);
//      auto patch_struct_name{fmt::format("init_cond_{}",l)};
      auto &patch_cond = parameter.get_struct(patch_struct_name);
      xs.push_back(std::get<real>(patch_cond.at("x0")));
      xe.push_back(std::get<real>(patch_cond.at("x1")));
      ys.push_back(std::get<real>(patch_cond.at("y0")));
      ye.push_back(std::get<real>(patch_cond.at("y1")));
      zs.push_back(std::get<real>(patch_cond.at("z0")));
      ze.push_back(std::get<real>(patch_cond.at("z1")));
      if (patch_cond.find("label") != patch_cond.cend()) {
        groups_inflow.emplace_back(parameter.get_struct(std::get<std::string>(patch_cond.at("label"))), species);
      } else {
        groups_inflow.emplace_back(parameter.get_struct(patch_struct_name), species);
      }
    }
  }

  // Start to initialize
  for (int blk = 0; blk < mesh.n_block; ++blk) {
    field[blk].initialize_basic_variables(parameter, groups_inflow, xs, xe, ys, ye, zs, ze);
  }


  if (parameter.get_int("myid") == 0) {
    printf("Flowfield is initialized from given inflow conditions.\n");
    std::ofstream history("history.dat", std::ios::trunc);
    history << "step\terror_max\n";
    history.close();
  }
}

template<MixtureModel mix_model, TurbMethod turb_method>
void
read_flowfield(cfd::Parameter &parameter, const cfd::Mesh &mesh, std::vector<Field<mix_model, turb_method>> &field,
               Species &species) {
  const std::filesystem::path out_dir("output/field");
  if (!exists(out_dir)) {
    printf("The directory to flowfield files does not exist!\n");
  }
  FILE *fp = fopen((out_dir.string() + std::format("/flowfield{:>4}.plt", parameter.get_int("myid"))).c_str(), "rb");

  std::string magic_number;
  fread(magic_number.data(), 8, 1, fp);
  int32_t byte_order{1};
  fread(&byte_order, 4, 1, fp);
  int32_t file_type{0};
  fread(&file_type, 4, 1, fp);
  std::string solution_file = gxl::read_str(fp);
  integer n_var_old{5};
  fread(&n_var_old, 4, 1, fp);
  std::vector<std::string> var_name;
  var_name.resize(n_var_old);
  for (size_t i = 0; i < n_var_old; ++i) {
    var_name[i] = gxl::read_str(fp);
  }
  // The first one tells if species info exists, if exists (1), else, (0).
  // The 2nd one tells if turbulent var exists, if 0 (compute from laminar), 1(From SA), 2(From SST)
  std::array old_data_info{0,0};//,0
  auto index_order = cfd::identify_variable_labels<mix_model, turb_method>(var_name, species, parameter,
                                                                           old_data_info);
  const integer n_spec{species.n_spec};

  float marker{0.0f};
  constexpr float eoh_marker{357.0f};
  fread(&marker, 4, 1, fp);
  std::vector<std::string> zone_name;
  std::vector<double> solution_time;
  integer zone_number{0};
  while (fabs(marker - eoh_marker) > 1e-25f) {
    zone_name.emplace_back(gxl::read_str(fp));
    int32_t parent_zone{-1};
    fread(&parent_zone, 4, 1, fp);
    int32_t strand_id{-2};
    fread(&strand_id, 4, 1, fp);
    real sol_time{0};
    fread(&sol_time, 8, 1, fp);
    solution_time.emplace_back(sol_time);
    int32_t zone_color{-1};
    fread(&zone_color, 4, 1, fp);
    int32_t zone_type{0};
    fread(&zone_type, 4, 1, fp);
    int32_t var_location{0};
    fread(&var_location, 4, 1, fp);
    int32_t raw_face_neighbor{0};
    fread(&raw_face_neighbor, 4, 1, fp);
    int32_t miscellaneous_face{0};
    fread(&miscellaneous_face, 4, 1, fp);
    integer mx{0}, my{0}, mz{0};
    fread(&mx, 4, 1, fp);
    fread(&my, 4, 1, fp);
    fread(&mz, 4, 1, fp);
    int32_t auxi_data{1};
    fread(&auxi_data, 4, 1, fp);
    while (auxi_data != 0) {
      auto auxi_name{gxl::read_str(fp)};
      int32_t auxi_format{0};
      fread(&auxi_format, 4, 1, fp);
      auto auxi_val{gxl::read_str(fp)};
      if (auxi_name == "step") {
        parameter.update_parameter("step", std::stoi(auxi_val));
      }
      fread(&auxi_data, 4, 1, fp);
    }
    ++zone_number;
    fread(&marker, 4, 1, fp);
  }

  // Next, data section
  for (size_t b = 0; b < mesh.n_block; ++b) {
    fread(&marker, 4, 1, fp);
    int32_t data_format{1};
    for (int l = 0; l < n_var_old; ++l) {
      fread(&data_format, 4, 1, fp);
    }
    size_t data_size{4};
    if (data_format == 2) {
      data_size = 8;
    }
    int32_t passive_var{0};
    fread(&passive_var, 4, 1, fp);
    int32_t shared_var{0};
    fread(&shared_var, 4, 1, fp);
    int32_t shared_connect{-1};
    fread(&shared_connect, 4, 1, fp);
    double max{0}, min{0};
    for (int l = 0; l < n_var_old; ++l) {
      fread(&min, 8, 1, fp);
      fread(&max, 8, 1, fp);
    }
    // zone data
    // First, the coordinates x, y and z.
    const integer mx{mesh[b].mx}, my{mesh[b].my}, mz{mesh[b].mz};
    for (size_t l = 0; l < 3; ++l) {
      read_one_useless_variable(fp, mx, my, mz, data_format);
    }

    // Other variables
    for (size_t l = 3; l < n_var_old; ++l) {
      auto index = index_order[l];
      if (index < 6) {
        // basic variables
        auto &bv = field[b].bv;
        if (data_format == 1) {
          // float storage
          float v{0.0f};
          for (int k = 0; k < mz; ++k) {
            for (int j = 0; j < my; ++j) {
              for (int i = 0; i < mx; ++i) {
                fread(&v, data_size, 1, fp);
                bv(i, j, k, index) = v;
              }
            }
          }
        } else {
          // double storage
          for (int k = 0; k < mz; ++k) {
            for (int j = 0; j < my; ++j) {
              for (int i = 0; i < mx; ++i) {
                fread(&bv(i, j, k, index), data_size, 1, fp);
              }
            }
          }
        }
      } else if (index < 6 + n_spec) {
        // If air, n_spec=0, this part will not cause any influence
        // species variables
        auto &yk = field[b].sv;
        index -= 6;
        if (data_format == 1) {
          // float storage
          float v{0.0f};
          for (int k = 0; k < mz; ++k) {
            for (int j = 0; j < my; ++j) {
              for (int i = 0; i < mx; ++i) {
                fread(&v, data_size, 1, fp);
                yk(i, j, k, index) = v;
              }
            }
          }
        } else {
          // double storage
          for (int k = 0; k < mz; ++k) {
            for (int j = 0; j < my; ++j) {
              for (int i = 0; i < mx; ++i) {
                fread(&yk(i, j, k, index), data_size, 1, fp);
              }
            }
          }
        }
      } else if (index == 6 + n_spec) {
        // Other variables, such as, turbulent variables et, al.
        // Each one has a independent label...
      } else {
        // No matched label, just ignore
        read_one_useless_variable(fp, mx, my, mz, data_format);
      }
    }
  }

  // Next, if the previous simulation does not contain some of the variables used in the current simulation,
  // then we initialize them here
  if constexpr (mix_model != MixtureModel::Air) {
//    constexpr integer index_spec_start{6};
//    const integer index_spec_end{6 + n_spec};
//    bool has_spec_info{false};
//    for (auto ii: index_order) {
//      if (ii >= index_spec_start && ii < index_spec_end) {
//        has_spec_info = true;
//      }
//    }
    if (old_data_info[0]==0) {
      initialize_spec_from_inflow(parameter, mesh, field, species);
    }
  }

  if (parameter.get_int("myid") == 0) {
    printf("Flowfield is initialized from previous simulation results.\n");
  }
}

template<MixtureModel mix_model, TurbMethod turb_method>
std::vector<integer>
identify_variable_labels(std::vector<std::string> &var_name, Species &species, Parameter &parameter,
                         std::array<integer, 2> &old_data_info) {
  std::vector<integer> labels;
  const integer n_spec = species.n_spec;
  for (auto &name: var_name) {
    integer l = 999;
    // The first three names are x, y and z, they are assigned value 0 and no match would be found.
    auto n = gxl::to_upper(name);
    if (n == "DENSITY" || n == "ROE" || n == "RHO") {
      l = 0;
    } else if (n == "U") {
      l = 1;
    } else if (n == "V") {
      l = 2;
    } else if (n == "W") {
      l = 3;
    } else if (n == "P" || n == "PRESSURE") {
      l = 4;
    } else if (n == "T" || n == "TEMPERATURE") {
      l = 5;
    } /*else if (n == "MUT") { // To be determined
      l = 6 + n_spec;
    }  else if (n == "MIXTURE FRACTION") { // mixture fraction
      l = 6 + n_spec + 2;
    } else if (n == "Z PRIME") { // fluctuation of mixture fraction
      l = 6 + n_spec + 3;
    }*/else {
      if constexpr (mix_model!=MixtureModel::Air){
        // We expect to find some species info. If not found, old_data_info[0] will remain 0.
        const auto &spec_name = species.spec_list;
        for (auto [spec, sp_label]: spec_name) {
          if (n == gxl::to_upper(spec)) {
            l = 6 + sp_label;
            old_data_info[0]=1;
            break;
          }
        }
      }
      if constexpr (turb_method==TurbMethod::RANS){
        // We expect to find some RANS variables. If not found, old_data_info[1] will remain 0.
        if (n == "K"){ // turbulent kinetic energy
          l = 6 + n_spec;
          old_data_info[1]=2; // SST model in previous simulation
        }else if (n == "OMEGA") { // specific dissipation rate
          l = 6 + n_spec + 1;
          old_data_info[1]=2; // SST model in previous simulation
        } else if (n == "NUT SA") { // the variable from SA, not named yet!!!
          l = 6 + n_spec;
          old_data_info[1]=1; // SA model in previous simulation
        }
      }
    }

    labels.emplace_back(l);
  }
  return labels;
}

template<MixtureModel mix_model, TurbMethod turb_method>
void initialize_spec_from_inflow(cfd::Parameter &parameter, const cfd::Mesh &mesh,
                                 std::vector<Field<mix_model, turb_method>> &field, Species &species) {
  // This can also be implemented like the from_start one, which can have patch.
  // But currently, for easy to implement, just intialize the whole flowfield to the inflow composition,
  // which means that other species would have to be computed from boundary conditions.
  // If the need for initialize species in groups is strong,
  // then we implement it just by copying the previous function "initialize_from_start",
  // which should be easy.
  const std::string default_init = parameter.get_string("default_init");
  Inflow<mix_model, turb_method> inflow(parameter.get_struct(default_init), species);
  for (int blk = 0; blk < mesh.n_block; ++blk) {
    const integer mx{mesh[blk].mx}, my{mesh[blk].my}, mz{mesh[blk].mz};
    const auto n_spec = parameter.get_int("n_spec");
    auto mass_frac = inflow.yk;
    auto &yk = field[blk].sv;
    for (int k = 0; k < mz; ++k) {
      for (int j = 0; j < my; ++j) {
        for (int i = 0; i < mx; ++i) {
          for (int l = 0; l < n_spec; ++l) {
            yk(i, j, k, l) = mass_frac[l];
          }
        }
      }
    }
  }
  if (parameter.get_int("myid") == 0) {
    printf("Compute from single species result. The species field is initialized with freestream.\n");
  }
}
}