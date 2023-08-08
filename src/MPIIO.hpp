#pragma once

#include "Define.h"
#include "mpi.h"
#include "Field.h"
#include "ChemData.h"
#include <filesystem>
#include <fstream>

namespace cfd {

void write_str(const char *str, MPI_File &file, MPI_Offset &offset);

void write_str_without_null(const char *str, MPI_File &file, MPI_Offset &offset);

template<MixtureModel mix_model, TurbMethod turb_method>
class MPIIO {
  const int myid{0};
  const Mesh &mesh;
  std::vector<Field> &field;
  const Parameter &parameter;
  const Species &species;
  int32_t n_var = 10;
  int ngg_output = 0;
  MPI_Offset offset_header = 0;
  MPI_Offset *offset_minmax_var = nullptr;
  MPI_Offset *offset_var = nullptr;

public:
  MPIIO(integer _myid, const Mesh &_mesh, std::vector<Field> &_field, const Parameter &_parameter, const Species &spec,
        int ngg_out);

  void print_field(integer step) const;

private:
  void write_header();

  void compute_offset_header();

  void write_common_data_section();

  int32_t acquire_variable_names(std::vector<std::string> &var_name) const;
};

template<MixtureModel mix_model, TurbMethod turb_method>
MPIIO<mix_model, turb_method>::MPIIO(integer _myid, const cfd::Mesh &_mesh, std::vector<Field> &_field,
                                     const cfd::Parameter &_parameter, const cfd::Species &spec,
                                     int ngg_out):
    myid{_myid}, mesh{_mesh}, field(_field), parameter{_parameter}, species{spec}, ngg_output{ngg_out} {
  const std::filesystem::path out_dir("output/field");
  if (!exists(out_dir)) {
    create_directories(out_dir);
  }
  write_header();
  compute_offset_header();
  write_common_data_section();
}

template<MixtureModel mix_model, TurbMethod turb_method>
void MPIIO<mix_model, turb_method>::write_header() {
  const std::filesystem::path out_dir("output/field");
  MPI_File fp;
  // Question: Should I use MPI_MODE_CREATE only here?
  // If a previous simulation has a larger file size than the current one, the way we write to the file is offset,
  // then the larger part would not be accessed anymore, which may results in a mistake.
  MPI_File_open(MPI_COMM_WORLD, (out_dir.string() + "/flowfield.plt").c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY,
                MPI_INFO_NULL, &fp);
  MPI_Status status;

  // I. Header section

  // Each file should have only one header, thus we let process 0 to write it.

  MPI_Offset offset{0};
  if (myid == 0) {
    // i. Magic number, Version number
    // V112 / V191. V112 was introduced in 2009 while V191 in 2019. They are different only in poly data, so no
    // difference is related to us. For common use, we use V112.
    constexpr auto magic_number{"#!TDV112"};
    write_str_without_null(magic_number, fp, offset);

    // ii. Integer value of 1
    constexpr int32_t byte_order{1};
    MPI_File_write_at(fp, offset, &byte_order, 1, MPI_INT32_T, &status);
    offset += 4;

    // iii. Title and variable names.
    // 1. FileType: 0=full, 1=grid, 2=solution
    constexpr int32_t file_type{0};
    MPI_File_write_at(fp, offset, &file_type, 1, MPI_INT32_T, &status);
    offset += 4;
    // 2. Title
    write_str("Solution file", fp, offset);
    // 3. Number of variables in the datafile, for this file, n_var = 3(x,y,z)+7(density,u,v,w,p,t,Ma)+n_spec+n_scalar
    std::vector<std::string> var_name{"x", "y", "z", "density", "u", "v", "w", "pressure", "temperature", "mach"};
    n_var = acquire_variable_names(var_name);
    MPI_File_write_at(fp, offset, &n_var, 1, MPI_INT32_T, &status);
    offset += 4;
    // 4. Variable names.
    for (auto &name: var_name) {
      write_str(name.c_str(), fp, offset);
    }

    // iv. Zones
    for (int i = 0; i < mesh.n_block_total; ++i) {
      // 1. Zone marker. Value = 299.0, indicates a V112 header.
      constexpr float zone_marker{299.0f};
      MPI_File_write_at(fp, offset, &zone_marker, 1, MPI_FLOAT, &status);
      offset += 4;
      // 2. Zone name.
      write_str(("zone " + std::to_string(i)).c_str(), fp, offset);
      // 3. Parent zone. No longer used
      constexpr int32_t parent_zone{-1};
      MPI_File_write_at(fp, offset, &parent_zone, 1, MPI_INT32_T, &status);
      offset += 4;
      // 4. Strand ID. -2 = pending strand ID for assignment by Tecplot; -1 = static strand ID; >= 0 valid strand ID
      constexpr int32_t strand_id{-2};
      MPI_File_write_at(fp, offset, &strand_id, 1, MPI_INT32_T, &status);
      offset += 4;
      // 5. Solution time. For steady, the value is set 0. For unsteady, please create a new class
      constexpr double solution_time{0};
      MPI_File_write_at(fp, offset, &solution_time, 1, MPI_DOUBLE, &status);
      offset += 8;
      // 6. Default Zone Color. Seldom used. Set to -1.
      constexpr int32_t zone_color{-1};
      MPI_File_write_at(fp, offset, &zone_color, 1, MPI_INT32_T, &status);
      offset += 4;
      // 7. ZoneType 0=ORDERED
      constexpr int32_t zone_type{0};
      MPI_File_write_at(fp, offset, &zone_type, 1, MPI_INT32_T, &status);
      offset += 4;
      // 8. Specify Var Location. 0 = All data is located at the nodes
      constexpr int32_t var_location{0};
      MPI_File_write_at(fp, offset, &var_location, 1, MPI_INT32_T, &status);
      offset += 4;
      // 9. Are raw local 1-to-1 face neighbors supplied? ORDERED zones must specify 0 for this value because
      // raw face neighbors are not defined for these zone types.
      constexpr int32_t raw_face_neighbor{0};
      MPI_File_write_at(fp, offset, &raw_face_neighbor, 1, MPI_INT32_T, &status);
      offset += 4;
      // 10. Number of miscellaneous user-defined face neighbor connections (value >= 0)
      constexpr int32_t miscellaneous_face{0};
      MPI_File_write_at(fp, offset, &miscellaneous_face, 1, MPI_INT32_T, &status);
      offset += 4;
      // For ordered zone, specify IMax, JMax, KMax
      const auto mx{mesh.mx_blk[i] + 2 * ngg_output}, my{mesh.my_blk[i] + 2 * ngg_output}, mz{
          mesh.mz_blk[i] + 2 * ngg_output};
      MPI_File_write_at(fp, offset, &mx, 1, MPI_INT32_T, &status);
      offset += 4;
      MPI_File_write_at(fp, offset, &my, 1, MPI_INT32_T, &status);
      offset += 4;
      MPI_File_write_at(fp, offset, &mz, 1, MPI_INT32_T, &status);
      offset += 4;

      // 11. For all zone types (repeat for each Auxiliary data name/value pair)
      // 1=Auxiliary name/value pair to follow; 0=No more Auxiliary name/value pairs.
      // If the above is 1, then supply the following: name string, Auxiliary Value Format, Value string
      // No more data
      constexpr int32_t no_more_auxi_data{0};
      MPI_File_write_at(fp, offset, &no_more_auxi_data, 1, MPI_INT32_T, &status);
      offset += 4;
    }

    // End of Header
    constexpr float EOHMARKER{357.0f};
    MPI_File_write_at(fp, offset, &EOHMARKER, 1, MPI_FLOAT, &status);
    offset += 4;

    offset_header = offset;
  }
  MPI_Bcast(&offset_header, 1, MPI_INT64_T, 0, MPI_COMM_WORLD);
  MPI_Bcast(&n_var, 1, MPI_INT32_T, 0, MPI_COMM_WORLD);
  MPI_File_close(&fp);
}

template<MixtureModel mix_model, TurbMethod turb_method>
void MPIIO<mix_model, turb_method>::compute_offset_header() {
  MPI_Offset new_offset{0};
  integer i_blk{0};
  for (int p = 0; p < myid; ++p) {
    const integer n_blk = mesh.nblk[p];
    for (int b = 0; b < n_blk; ++b) {
      new_offset += 16 + 20 * n_var;
      const integer mx{mesh.mx_blk[i_blk] + 2 * ngg_output}, my{mesh.my_blk[i_blk] + 2 * ngg_output}, mz{
          mesh.mz_blk[i_blk] + 2 * ngg_output};
      const integer N = mx * my * mz;
      // We always write double precision out
      new_offset += n_var * N * 8;
      ++i_blk;
    }
  }
  offset_header += new_offset;
}

template<MixtureModel mix_model, TurbMethod turb_method>
void MPIIO<mix_model, turb_method>::write_common_data_section() {
  const std::filesystem::path out_dir("output/field");
  MPI_File fp;
  MPI_File_open(MPI_COMM_WORLD, (out_dir.string() + "/flowfield.plt").c_str(), MPI_MODE_WRONLY,
                MPI_INFO_NULL, &fp);
  MPI_Status status;

  const auto n_block{mesh.n_block};
  offset_minmax_var = new MPI_Offset[n_block];
  offset_var = new MPI_Offset[n_block];

  auto offset{offset_header};
  const auto ngg{ngg_output};
  for (int blk = 0; blk < n_block; ++blk) {
    // 1. Zone marker. Value = 299.0, indicates a V112 header.
    constexpr float zone_marker{299.0f};
    MPI_File_write_at(fp, offset, &zone_marker, 1, MPI_FLOAT, &status);
    offset += 4;
    // 2. Variable data format, 1=Float, 2=Double, 3=LongInt, 4=ShortInt, 5=Byte, 6=Bit
    constexpr int32_t data_format{2};
    for (int l = 0; l < n_var; ++l) {
      MPI_File_write_at(fp, offset, &data_format, 1, MPI_INT32_T, &status);
      offset += 4;
    }
    // 3. Has passive variables: 0 = no, 1 = yes.
    constexpr int32_t passive_var{0};
    MPI_File_write_at(fp, offset, &passive_var, 1, MPI_INT32_T, &status);
    offset += 4;
    // 4. Has variable sharing 0 = no, 1 = yes.
    constexpr int32_t shared_var{0};
    MPI_File_write_at(fp, offset, &shared_var, 1, MPI_INT32_T, &status);
    offset += 4;
    // 5. Zero based zone number to share connectivity list with (-1 = no sharing).
    constexpr int32_t shared_connect{-1};
    MPI_File_write_at(fp, offset, &shared_connect, 1, MPI_INT32_T, &status);
    offset += 4;
    // 6. Compressed list of min/max pairs for each non-shared and non-passive variable.
    // For each non-shared and non-passive variable (as specified above):
    auto &b{mesh[blk]};
    auto &v{field[blk]};
    const auto mx{b.mx}, my{b.my}, mz{b.mz};

    double min_val{b.x(-ngg, -ngg, -ngg)}, max_val{b.x(-ngg, -ngg, -ngg)};
    for (int k = -ngg; k < mz + ngg; ++k) {
      for (int j = -ngg; j < my + ngg; ++j) {
        for (int i = -ngg; i < mx + ngg; ++i) {
          min_val = std::min(min_val, b.x(i, j, k));
          max_val = std::max(max_val, b.x(i, j, k));
        }
      }
    }
    MPI_File_write_at(fp, offset, &min_val, 1, MPI_DOUBLE, &status);
    offset += 8;
    MPI_File_write_at(fp, offset, &max_val, 1, MPI_DOUBLE, &status);
    offset += 8;
    min_val = b.y(-ngg, -ngg, -ngg);
    max_val = b.y(-ngg, -ngg, -ngg);
    for (int k = -ngg; k < mz + ngg; ++k) {
      for (int j = -ngg; j < my + ngg; ++j) {
        for (int i = -ngg; i < mx + ngg; ++i) {
          min_val = std::min(min_val, b.y(i, j, k));
          max_val = std::max(max_val, b.y(i, j, k));
        }
      }
    }
    MPI_File_write_at(fp, offset, &min_val, 1, MPI_DOUBLE, &status);
    offset += 8;
    MPI_File_write_at(fp, offset, &max_val, 1, MPI_DOUBLE, &status);
    offset += 8;
    min_val = b.z(-ngg, -ngg, -ngg);
    max_val = b.z(-ngg, -ngg, -ngg);
    for (int k = -ngg; k < mz + ngg; ++k) {
      for (int j = -ngg; j < my + ngg; ++j) {
        for (int i = -ngg; i < mx + ngg; ++i) {
          min_val = std::min(min_val, b.z(i, j, k));
          max_val = std::max(max_val, b.z(i, j, k));
        }
      }
    }
    MPI_File_write_at(fp, offset, &min_val, 1, MPI_DOUBLE, &status);
    offset += 8;
    MPI_File_write_at(fp, offset, &max_val, 1, MPI_DOUBLE, &status);
    offset += 8;
    // Later, the max/min values of flow variables are computed.
    offset_minmax_var[blk] = offset;
    for (int l = 0; l < 6; ++l) {
      min_val = v.bv(-ngg, -ngg, -ngg, l);
      max_val = v.bv(-ngg, -ngg, -ngg, l);
      for (int k = -ngg; k < mz + ngg; ++k) {
        for (int j = -ngg; j < my + ngg; ++j) {
          for (int i = -ngg; i < mx + ngg; ++i) {
            min_val = std::min(min_val, v.bv(i, j, k, l));
            max_val = std::max(max_val, v.bv(i, j, k, l));
          }
        }
      }
      MPI_File_write_at(fp, offset, &min_val, 1, MPI_DOUBLE, &status);
      offset += 8;
      MPI_File_write_at(fp, offset, &max_val, 1, MPI_DOUBLE, &status);
      offset += 8;
    }
    min_val = v.ov(-ngg, -ngg, -ngg, 0);
    max_val = v.ov(-ngg, -ngg, -ngg, 0);
    for (int k = -ngg; k < mz + ngg; ++k) {
      for (int j = -ngg; j < my + ngg; ++j) {
        for (int i = -ngg; i < mx + ngg; ++i) {
          min_val = std::min(min_val, v.ov(i, j, k, 0));
          max_val = std::max(max_val, v.ov(i, j, k, 0));
        }
      }
    }
    MPI_File_write_at(fp, offset, &min_val, 1, MPI_DOUBLE, &status);
    offset += 8;
    MPI_File_write_at(fp, offset, &max_val, 1, MPI_DOUBLE, &status);
    offset += 8;
    // scalar variables. Y0-Y_{Ns-1}, k, omega, z, z_prime
    const integer n_scalar{parameter.get_int("n_scalar")};
    for (int l = 0; l < n_scalar; ++l) {
      min_val = v.sv(-ngg, -ngg, -ngg, l);
      max_val = v.sv(-ngg, -ngg, -ngg, l);
      for (int k = -ngg; k < mz + ngg; ++k) {
        for (int j = -ngg; j < my + ngg; ++j) {
          for (int i = -ngg; i < mx + ngg; ++i) {
            min_val = std::min(min_val, v.sv(i, j, k, l));
            max_val = std::max(max_val, v.sv(i, j, k, l));
          }
        }
      }
      MPI_File_write_at(fp, offset, &min_val, 1, MPI_DOUBLE, &status);
      offset += 8;
      MPI_File_write_at(fp, offset, &max_val, 1, MPI_DOUBLE, &status);
      offset += 8;
    }
    // if turbulent, mut
    if constexpr (turb_method == TurbMethod::RANS || turb_method == TurbMethod::LES) {
      min_val = v.ov(-ngg, -ngg, -ngg, 1);
      max_val = v.ov(-ngg, -ngg, -ngg, 1);
      for (int k = -ngg; k < mz + ngg; ++k) {
        for (int j = -ngg; j < my + ngg; ++j) {
          for (int i = -ngg; i < mx + ngg; ++i) {
            min_val = std::min(min_val, v.ov(i, j, k, 1));
            max_val = std::max(max_val, v.ov(i, j, k, 1));
          }
        }
      }
      MPI_File_write_at(fp, offset, &min_val, 1, MPI_DOUBLE, &status);
      offset += 8;
      MPI_File_write_at(fp, offset, &max_val, 1, MPI_DOUBLE, &status);
      offset += 8;
    }

    // 7. Zone Data.
    MPI_Datatype ty;
    integer lsize[3]{mx + 2 * ngg, my + 2 * ngg, mz + 2 * ngg};
    const auto memsz = lsize[0] * lsize[1] * lsize[2] * 8;
    integer memsize[3]{mx + 2 * b.ngg, my + 2 * b.ngg, mz + 2 * b.ngg};
    integer start_idx[3]{b.ngg - ngg, b.ngg - ngg, b.ngg - ngg};
    MPI_Type_create_subarray(3, memsize, lsize, start_idx, MPI_ORDER_FORTRAN, MPI_DOUBLE, &ty);
    MPI_Type_commit(&ty);
    MPI_File_write_at(fp, offset, b.x.data(), 1, ty, &status);
    offset += memsz;
    MPI_File_write_at(fp, offset, b.y.data(), 1, ty, &status);
    offset += memsz;
    MPI_File_write_at(fp, offset, b.z.data(), 1, ty, &status);
    offset += memsz;
    // Later, the variables are outputted.
    offset_var[blk] = offset;
    for (int l = 0; l < 6; ++l) {
      auto var = v.bv[l];
      MPI_File_write_at(fp, offset, var, 1, ty, &status);
      offset += memsz;
    }
    auto var = v.ov[0];
    MPI_File_write_at(fp, offset, var, 1, ty, &status);
    offset += memsz;
    for (int l = 0; l < field[0].n_var - 5; ++l) {
      var = v.sv[l];
      MPI_File_write_at(fp, offset, var, 1, ty, &status);
      offset += memsz;
    }
    // if turbulent, mut
    if constexpr (turb_method == TurbMethod::RANS || turb_method == TurbMethod::LES) {
      var = v.ov[1];
      MPI_File_write_at(fp, offset, var, 1, ty, &status);
      offset += memsz;
    }
  }

  MPI_Barrier(MPI_COMM_WORLD);
  MPI_File_close(&fp);
}

template<MixtureModel mix_model, TurbMethod turb_method>
int32_t MPIIO<mix_model, turb_method>::acquire_variable_names(std::vector<std::string> &var_name) const {
  int32_t nv = 3 + 7; // x,y,z + rho,u,v,w,p,T,Mach
  if constexpr (mix_model != MixtureModel::Air) {
    nv += parameter.get_int("n_spec"); // Y_k
    var_name.resize(nv);
    auto &names = species.spec_list;
    for (auto &[name, ind]: names) {
      var_name[ind + 10] = name;
    }
  }
  if constexpr (turb_method == TurbMethod::RANS) {
    if (integer rans_method = parameter.get_int("RANS_model"); rans_method == 1) {
      nv += 1; // SA variable?
    } else if (rans_method == 2) {
      nv += 2; // k, omega
      var_name.emplace_back("tke");
      var_name.emplace_back("omega");
    }
  }
  if constexpr (mix_model == MixtureModel::FL) {
    nv += 2; // Z, Z_prime
    var_name.emplace_back("z");
    var_name.emplace_back("z prime");
  }
  if constexpr (turb_method == TurbMethod::RANS || turb_method == TurbMethod::LES) {
    nv += 1; // mu_t
    var_name.emplace_back("mut");
  }
  return nv;
}

template<MixtureModel mix_model, TurbMethod turb_method>
void MPIIO<mix_model, turb_method>::print_field(integer step) const {
  if (myid == 0) {
    std::ofstream file("output/message/step.txt");
    file << step;
    file.close();
  }

  // Copy data from GPU to CPU
  for (auto &f: field) {
    f.copy_data_from_device(parameter);
  }

  const std::filesystem::path out_dir("output/field");
  MPI_File fp;
  MPI_File_open(MPI_COMM_WORLD, (out_dir.string() + "/flowfield.plt").c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY,
                MPI_INFO_NULL, &fp);
  MPI_Status status;

  // II. Data Section
  // First, modify the new min/max values of the variables
  for (int blk = 0; blk < mesh.n_block; ++blk) {
    MPI_Offset offset{offset_minmax_var[blk]};

    double min_val{0}, max_val{1};
    auto &b{mesh[blk]};
    auto &v{field[blk]};
    const auto mx{b.mx}, my{b.my}, mz{b.mz};
    const auto ngg{ngg_output};
    for (int l = 0; l < 6; ++l) {
      min_val = v.bv(-ngg, -ngg, -ngg, l);
      max_val = v.bv(-ngg, -ngg, -ngg, l);
      for (int k = -ngg; k < mz + ngg; ++k) {
        for (int j = -ngg; j < my + ngg; ++j) {
          for (int i = -ngg; i < mx + ngg; ++i) {
            min_val = std::min(min_val, v.bv(i, j, k, l));
            max_val = std::max(max_val, v.bv(i, j, k, l));
          }
        }
      }
      MPI_File_write_at(fp, offset, &min_val, 1, MPI_DOUBLE, &status);
      offset += 8;
      MPI_File_write_at(fp, offset, &max_val, 1, MPI_DOUBLE, &status);
      offset += 8;
    }
    min_val = v.ov(-ngg, -ngg, -ngg, 0);
    max_val = v.ov(-ngg, -ngg, -ngg, 0);
    for (int k = -ngg; k < mz + ngg; ++k) {
      for (int j = -ngg; j < my + ngg; ++j) {
        for (int i = -ngg; i < mx + ngg; ++i) {
          min_val = std::min(min_val, v.ov(i, j, k, 0));
          max_val = std::max(max_val, v.ov(i, j, k, 0));
        }
      }
    }
    MPI_File_write_at(fp, offset, &min_val, 1, MPI_DOUBLE, &status);
    offset += 8;
    MPI_File_write_at(fp, offset, &max_val, 1, MPI_DOUBLE, &status);
    offset += 8;
    // scalar variables. Y0-Y_{Ns-1}, k, omega, z, z_prime
    const integer n_scalar{parameter.get_int("n_scalar")};
    for (int l = 0; l < n_scalar; ++l) {
      min_val = v.sv(-ngg, -ngg, -ngg, l);
      max_val = v.sv(-ngg, -ngg, -ngg, l);
      for (int k = -ngg; k < mz + ngg; ++k) {
        for (int j = -ngg; j < my + ngg; ++j) {
          for (int i = -ngg; i < mx + ngg; ++i) {
            min_val = std::min(min_val, v.sv(i, j, k, l));
            max_val = std::max(max_val, v.sv(i, j, k, l));
          }
        }
      }
      MPI_File_write_at(fp, offset, &min_val, 1, MPI_DOUBLE, &status);
      offset += 8;
      MPI_File_write_at(fp, offset, &max_val, 1, MPI_DOUBLE, &status);
      offset += 8;
    }
    // if turbulent, mut
    if constexpr (turb_method == TurbMethod::RANS || turb_method == TurbMethod::LES) {
      min_val = v.ov(-ngg, -ngg, -ngg, 1);
      max_val = v.ov(-ngg, -ngg, -ngg, 1);
      for (int k = -ngg; k < mz + ngg; ++k) {
        for (int j = -ngg; j < my + ngg; ++j) {
          for (int i = -ngg; i < mx + ngg; ++i) {
            min_val = std::min(min_val, v.ov(i, j, k, 1));
            max_val = std::max(max_val, v.ov(i, j, k, 1));
          }
        }
      }
      MPI_File_write_at(fp, offset, &min_val, 1, MPI_DOUBLE, &status);
      offset += 8;
      MPI_File_write_at(fp, offset, &max_val, 1, MPI_DOUBLE, &status);
      offset += 8;
    }

    // 7. Zone Data.
    MPI_Datatype ty;
    integer lsize[3]{mx + 2 * ngg, my + 2 * ngg, mz + 2 * ngg};
    const auto memsz = lsize[0] * lsize[1] * lsize[2] * 8;
    integer memsize[3]{mx + 2 * b.ngg, my + 2 * b.ngg, mz + 2 * b.ngg};
    integer start_idx[3]{b.ngg - ngg, b.ngg - ngg, b.ngg - ngg};
    MPI_Type_create_subarray(3, memsize, lsize, start_idx, MPI_ORDER_FORTRAN, MPI_DOUBLE, &ty);
    MPI_Type_commit(&ty);

    offset = offset_var[blk];
    for (int l = 0; l < 6; ++l) {
      auto var = v.bv[l];
      MPI_File_write_at(fp, offset, var, 1, ty, &status);
      offset += memsz;
    }
    auto var = v.ov[0];
    MPI_File_write_at(fp, offset, var, 1, ty, &status);
    offset += memsz;
    for (int l = 0; l < field[0].n_var - 5; ++l) {
      var = v.sv[l];
      MPI_File_write_at(fp, offset, var, 1, ty, &status);
      offset += memsz;
    }
    // if turbulent, mut
    if constexpr (turb_method == TurbMethod::RANS || turb_method == TurbMethod::LES) {
      var = v.ov[1];
      MPI_File_write_at(fp, offset, var, 1, ty, &status);
      offset += memsz;
    }
  }
  MPI_File_close(&fp);
}

}
