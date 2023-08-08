#pragma once

#include <vector>
#include "Parameter.h"
#include "Field.h"
#include <filesystem>
#include "ChemData.h"
#include "gxl_lib/MyString.h"

namespace cfd {
// Normally, the forward declaration should either be claimed as struct or class, but this time, the type must match
// , or it will not be able to find the corresponding libs.
class Mesh;


template<MixtureModel mix_model, TurbMethod turb_method>
class Output {
public:
  const int myid{0};
  const Mesh &mesh;
  std::vector<cfd::Field> &field;
  const Parameter &parameter;
  const Species &species;

  Output(integer _myid, const Mesh &_mesh, std::vector<Field> &_field, const Parameter &_parameter,
         const Species &spec);

  int32_t acquire_variable_names(std::vector<std::string> &var_name) const;

  void print_field(integer step, int ngg = 0) const;

  ~Output() = default;
};

template<MixtureModel mix_model, TurbMethod turb_method>
int32_t Output<mix_model, turb_method>::acquire_variable_names(std::vector<std::string> &var_name) const {
  int32_t n_var = 3 + 7; // x,y,z + rho,u,v,w,p,T,Mach
  if constexpr (mix_model != MixtureModel::Air) {
    n_var += parameter.get_int("n_spec"); // Y_k
    var_name.resize(n_var);
    auto &names = species.spec_list;
    for (auto &[name, ind]: names) {
      var_name[ind + 10] = name;
    }
  }
  if constexpr (turb_method == TurbMethod::RANS) {
    if (integer rans_method = parameter.get_int("RANS_model"); rans_method == 1) {
      n_var += 1; // SA variable?
    } else if (rans_method == 2) {
      n_var += 2; // k, omega
      var_name.emplace_back("tke");
      var_name.emplace_back("omega");
    }
  }
  if constexpr (mix_model == MixtureModel::FL) {
    n_var += 2; // Z, Z_prime
    var_name.emplace_back("z");
    var_name.emplace_back("z prime");
  }
  if constexpr (turb_method == TurbMethod::RANS || turb_method == TurbMethod::LES) {
    n_var += 1; // mu_t
    var_name.emplace_back("mut");
  }
  return n_var;
}

template<MixtureModel mix_model, TurbMethod turb_method>
void Output<mix_model, turb_method>::print_field(integer step, int ngg) const {
  // Copy data from GPU to CPU
  for (auto &f: field) {
    f.copy_data_from_device(parameter);
  }

  const std::filesystem::path out_dir("output/field");
  char id[5];
  sprintf(id, "%4d", myid);
  std::string id_str = id;
  FILE *fp = fopen((out_dir.string() + "/flowfield" + id_str + ".plt").c_str(), "wb");
//  FILE *fp = fopen((out_dir.string() + std::format("/flowfield{:>4}.plt", myid)).c_str(), "wb");

  // I. Header section

  // i. Magic number, Version number
  // V112 / V191. V112 was introduced in 2009 while V191 in 2019. They are different only in poly data, so no
  // difference is related to us. For common use, we use V112.
  constexpr auto magic_number{"#!TDV112"};
  fwrite(magic_number, 8, 1, fp);

  // ii. Integer value of 1
  constexpr int32_t byte_order{1};
  fwrite(&byte_order, 4, 1, fp);

  // iii. Title and variable names.
  // 1. FileType: 0=full, 1=grid, 2=solution
  constexpr int32_t file_type{0};
  fwrite(&file_type, 4, 1, fp);
  // 2. Title
  gxl::write_str("Solution file", fp);
  // 3. Number of variables in the datafile, for this file, n_var = 3(x,y,z)+7(density,u,v,w,p,t,Ma)+n_spec+n_scalar
  std::vector<std::string> var_name{"x", "y", "z", "density", "u", "v", "w", "pressure", "temperature", "mach"};
  int32_t n_var = acquire_variable_names(var_name);
  fwrite(&n_var, 4, 1, fp);
  // 4. Variable names.
  for (auto &name: var_name) {
    gxl::write_str(name.c_str(), fp);
  }

  // iv. Zones
  for (int i = 0; i < mesh.n_block; ++i) {
    // 1. Zone marker. Value = 299.0, indicates a V112 header.
    constexpr float zone_marker{299.0f};
    fwrite(&zone_marker, 4, 1, fp);
    // 2. Zone name.
    gxl::write_str(("zone " + std::to_string(i)).c_str(), fp);
    // 3. Parent zone. No longer used
    constexpr int32_t parent_zone{-1};
    fwrite(&parent_zone, 4, 1, fp);
    // 4. Strand ID. -2 = pending strand ID for assignment by Tecplot; -1 = static strand ID; >= 0 valid strand ID
    constexpr int32_t strand_id{-2};
    fwrite(&strand_id, 4, 1, fp);
    // 5. Solution time. For steady, the value is set 0. For unsteady, please create a new class
    constexpr real solution_time{0};
    fwrite(&solution_time, 8, 1, fp);
    // 6. Default Zone Color. Seldom used. Set to -1.
    constexpr int32_t zone_color{-1};
    fwrite(&zone_color, 4, 1, fp);
    // 7. ZoneType 0=ORDERED
    constexpr int32_t zone_type{0};
    fwrite(&zone_type, 4, 1, fp);
    // 8. Specify Var Location. 0 = All data is located at the nodes
    constexpr int32_t var_location{0};
    fwrite(&var_location, 4, 1, fp);
    // 9. Are raw local 1-to-1 face neighbors supplied? ORDERED zones must specify 0 for this value because
    // raw face neighbors are not defined for these zone types.
    constexpr int32_t raw_face_neighbor{0};
    fwrite(&raw_face_neighbor, 4, 1, fp);
    // 10. Number of miscellaneous user-defined face neighbor connections (value >= 0)
    constexpr int32_t miscellaneous_face{0};
    fwrite(&miscellaneous_face, 4, 1, fp);
    // For ordered zone, specify IMax, JMax, KMax
    auto &b = mesh[i];
    const auto mx{b.mx + 2 * ngg}, my{b.my + 2 * ngg}, mz{b.mz + 2 * ngg};
    fwrite(&mx, 4, 1, fp);
    fwrite(&my, 4, 1, fp);
    fwrite(&mz, 4, 1, fp);

    // 11. For all zone types (repeat for each Auxiliary data name/value pair)
    // 1=Auxiliary name/value pair to follow; 0=No more Auxiliary name/value pairs.
    // If the above is 1, then supply the following: name string, Auxiliary Value Format, Value string

    // First, record the current simulation step
    constexpr int32_t auxi_data{1};
    fwrite(&auxi_data, 4, 1, fp);
    // Name string
    constexpr auto step_name{"step"};
    gxl::write_str(step_name, fp);
    // Auxiliary Value Format(Currently only allow 0=AuxDataType_String)
    constexpr int32_t auxi_val_form{0};
    fwrite(&auxi_val_form, 4, 1, fp);
    // Value string
    const auto step_str = std::to_string(step);
    gxl::write_str(step_str.c_str(), fp);

    // No more data
    constexpr int32_t no_more_auxi_data{0};
    fwrite(&no_more_auxi_data, 4, 1, fp);
  }

  // End of Header
  constexpr float EOHMARKER{357.0f};
  fwrite(&EOHMARKER, 4, 1, fp);

  // II. Data Section
  for (int blk = 0; blk < mesh.n_block; ++blk) {
    // 1. Zone marker. Value = 299.0, indicates a V112 header.
    constexpr float zone_marker{299.0f};
    fwrite(&zone_marker, 4, 1, fp);
    // 2. Variable data format, 1=Float, 2=Double, 3=LongInt, 4=ShortInt, 5=Byte, 6=Bit
    constexpr int32_t data_format{1};
    constexpr size_t data_size{4};
    for (int l = 0; l < n_var; ++l) {
      fwrite(&data_format, 4, 1, fp);
    }
    // 3. Has passive variables: 0 = no, 1 = yes.
    constexpr int32_t passive_var{0};
    fwrite(&passive_var, 4, 1, fp);
    // 4. Has variable sharing 0 = no, 1 = yes.
    constexpr int32_t shared_var{0};
    fwrite(&shared_var, 4, 1, fp);
    // 5. Zero based zone number to share connectivity list with (-1 = no sharing).
    constexpr int32_t shared_connect{-1};
    fwrite(&shared_connect, 4, 1, fp);
    // 6. Compressed list of min/max pairs for each non-shared and non-passive variable.
    // For each non-shared and non-passive variable (as specified above):
    auto &b{mesh[blk]};
    auto &v{field[blk]};
    const auto mx{b.mx}, my{b.my}, mz{b.mz};
    const std::vector<gxl::Array3D<double>> &vars{b.x, b.y, b.z};
    // Potential optimization: the x/y/z coordinates are fixed, thus their max/min values can be saved instead of comparing them every time.
    for (auto &var: vars) {
      double min_val{var(-ngg, -ngg, -ngg)}, max_val{var(-ngg, -ngg, -ngg)};
      for (int k = -ngg; k < mz + ngg; ++k) {
        for (int j = -ngg; j < my + ngg; ++j) {
          for (int i = -ngg; i < mx + ngg; ++i) {
            min_val = std::min(min_val, var(i, j, k));
            max_val = std::max(max_val, var(i, j, k));
          }
        }
      }
      fwrite(&min_val, 8, 1, fp);
      fwrite(&max_val, 8, 1, fp);
    }
    std::array min_val{
        v.bv(-ngg, -ngg, -ngg, 0), v.bv(-ngg, -ngg, -ngg, 1), v.bv(-ngg, -ngg, -ngg, 2),
        v.bv(-ngg, -ngg, -ngg, 3), v.bv(-ngg, -ngg, -ngg, 4), v.bv(-ngg, -ngg, -ngg, 5)
    };
    std::array max_val{
        v.bv(-ngg, -ngg, -ngg, 0), v.bv(-ngg, -ngg, -ngg, 1), v.bv(-ngg, -ngg, -ngg, 2),
        v.bv(-ngg, -ngg, -ngg, 3), v.bv(-ngg, -ngg, -ngg, 4), v.bv(-ngg, -ngg, -ngg, 5)
    };
    for (int l = 0; l < 6; ++l) {
      for (int k = -ngg; k < mz + ngg; ++k) {
        for (int j = -ngg; j < my + ngg; ++j) {
          for (int i = -ngg; i < mx + ngg; ++i) {
            min_val[l] = std::min(min_val[l], v.bv(i, j, k, l));
            max_val[l] = std::max(max_val[l], v.bv(i, j, k, l));
          }
        }
      }
    }
    for (int l = 0; l < 6; ++l) {
      fwrite(&min_val[l], 8, 1, fp);
      fwrite(&max_val[l], 8, 1, fp);
    }
    min_val[0] = v.ov(-ngg, -ngg, -ngg, 0);
    max_val[0] = v.ov(-ngg, -ngg, -ngg, 0);
    for (int k = -ngg; k < mz + ngg; ++k) {
      for (int j = -ngg; j < my + ngg; ++j) {
        for (int i = -ngg; i < mx + ngg; ++i) {
          min_val[0] = std::min(min_val[0], v.ov(i, j, k, 0));
          max_val[0] = std::max(max_val[0], v.ov(i, j, k, 0));
        }
      }
    }
    fwrite(min_val.data(), 8, 1, fp);
    fwrite(max_val.data(), 8, 1, fp);
    // scalar variables. Y0-Y_{Ns-1}, k, omega, z, z_prime
    const integer n_scalar{field[0].h_ptr->n_scal};
    std::vector<double> s_min(n_scalar, 0), s_max(n_scalar, 0);
    for (int l = 0; l < n_scalar; ++l) {
      s_min[l] = v.sv(-ngg, -ngg, -ngg, l);
      s_max[l] = v.sv(-ngg, -ngg, -ngg, l);
      for (int k = -ngg; k < mz + ngg; ++k) {
        for (int j = -ngg; j < my + ngg; ++j) {
          for (int i = -ngg; i < mx + ngg; ++i) {
            s_min[l] = std::min(s_min[l], v.sv(i, j, k, l));
            s_max[l] = std::max(s_max[l], v.sv(i, j, k, l));
          }
        }
      }
    }
    for (int l = 0; l < n_scalar; ++l) {
      fwrite(&s_min[l], 8, 1, fp);
      fwrite(&s_max[l], 8, 1, fp);
    }
    // if turbulent, mut
    if constexpr (turb_method == TurbMethod::RANS || turb_method == TurbMethod::LES) {
      min_val[0] = v.ov(-ngg, -ngg, -ngg, 1);
      max_val[0] = v.ov(-ngg, -ngg, -ngg, 1);
      for (int k = -ngg; k < mz + ngg; ++k) {
        for (int j = -ngg; j < my + ngg; ++j) {
          for (int i = -ngg; i < mx + ngg; ++i) {
            min_val[0] = std::min(min_val[0], v.ov(i, j, k, 1));
            max_val[0] = std::max(max_val[0], v.ov(i, j, k, 1));
          }
        }
      }
      fwrite(min_val.data(), 8, 1, fp);
      fwrite(max_val.data(), 8, 1, fp);
    }

    // 7. Zone Data.
    for (auto &var: vars) {
      for (int k = -ngg; k < mz + ngg; ++k) {
        for (int j = -ngg; j < my + ngg; ++j) {
          for (int i = -ngg; i < mx + ngg; ++i) {
            const auto value = static_cast<float>(var(i, j, k));
            fwrite(&value, data_size, 1, fp);
          }
        }
      }
    }
    for (int l = 0; l < 6; ++l) {
      for (int k = -ngg; k < mz + ngg; ++k) {
        for (int j = -ngg; j < my + ngg; ++j) {
          for (int i = -ngg; i < mx + ngg; ++i) {
            const auto value = static_cast<float>(v.bv(i, j, k, l));
            fwrite(&value, data_size, 1, fp);
          }
        }
      }
    }
    for (int k = -ngg; k < mz + ngg; ++k) {
      for (int j = -ngg; j < my + ngg; ++j) {
        for (int i = -ngg; i < mx + ngg; ++i) {
          const auto value = static_cast<float>(v.ov(i, j, k, 0));
          fwrite(&value, data_size, 1, fp);
        }
      }
    }
    for (int l = 0; l < n_scalar; ++l) {
      for (int k = -ngg; k < mz + ngg; ++k) {
        for (int j = -ngg; j < my + ngg; ++j) {
          for (int i = -ngg; i < mx + ngg; ++i) {
            const auto value = static_cast<float>(v.sv(i, j, k, l));
            fwrite(&value, data_size, 1, fp);
          }
        }
      }
    }
    // if turbulent, mut
    if constexpr (turb_method == TurbMethod::RANS || turb_method == TurbMethod::LES) {
      for (int k = -ngg; k < mz + ngg; ++k) {
        for (int j = -ngg; j < my + ngg; ++j) {
          for (int i = -ngg; i < mx + ngg; ++i) {
            const auto value = static_cast<float>(v.ov(i, j, k, 1));
            fwrite(&value, data_size, 1, fp);
          }
        }
      }
    }
  }
  fclose(fp);
}

// Implementations
template<MixtureModel mix_model, TurbMethod turb_method>
Output<mix_model, turb_method>::Output(integer _myid, const cfd::Mesh &_mesh, std::vector<Field> &_field,
                                       const cfd::Parameter &_parameter, const cfd::Species &spec):
    myid{_myid}, mesh{_mesh}, field(_field), parameter{_parameter}, species{spec} {
  const std::filesystem::path out_dir("output/field");
  if (!exists(out_dir)) {
    create_directories(out_dir);
  }
}
} // cfd
