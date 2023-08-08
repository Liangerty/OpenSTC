#pragma once

#include "Parameter.h"
#include "Field.h"
#include "ChemData.h"
#include "BoundCond.h"
#include <filesystem>
#include <mpi.h>
#include "gxl_lib/MyString.h"

namespace cfd {
template<MixtureModel mix_model, TurbMethod turb_method>
void initialize_basic_variables(Parameter &parameter, const Mesh &mesh, std::vector<Field> &field, Species &species);

void initialize_from_start(Parameter &parameter, const Mesh &mesh, std::vector<Field> &field, Species &species);

template<MixtureModel mix_model, TurbMethod turb_method>
void read_flowfield(Parameter &parameter, const Mesh &mesh, std::vector<Field> &field, Species &species);

/**
 * @brief To relate the order of variables from the flowfield files to bv, yk, turbulent arrays
 * @param var_name the array which contains all variables from the flowfield files
 * @return an array of orders. 0~5 means density, u, v, w, p, T; 6~5+ns means the species order, 6+ns~... means other variables such as mut...
 */
template<MixtureModel mix_model, TurbMethod turb_method>
std::vector<integer>
identify_variable_labels(cfd::Parameter &parameter, std::vector<std::string> &var_name, Species &species,
                         std::array<integer, 2> &old_data_info);

void initialize_spec_from_inflow(cfd::Parameter &parameter, const cfd::Mesh &mesh,
                                 std::vector<Field> &field, Species &species);

void initialize_turb_from_inflow(cfd::Parameter &parameter, const cfd::Mesh &mesh,
                                 std::vector<Field> &field, Species &species);

// Implementations
template<MixtureModel mix_model, TurbMethod turb_method>
void initialize_basic_variables(Parameter &parameter, const Mesh &mesh, std::vector<Field> &field, Species &species) {
  const integer init_method = parameter.get_int("initial");
  // No matter which method is used to initialize the flowfield,
  // the default inflow is first read and initialize the inf parameters.
  // Otherwise, for simulations that begin from previous simulations,
  // processes other than the one containing the inflow plane would have no info about inf parameters.
  const std::string default_init = parameter.get_string("default_init");
  [[maybe_unused]] Inflow default_inflow(default_init, species, parameter);

  switch (init_method) {
    case 0:initialize_from_start(parameter, mesh, field, species);
      break;
    case 1:read_flowfield<mix_model, turb_method>(parameter, mesh, field, species);
      break;
    default:printf("The initialization method is unknown, use freestream value to initialize by default.\n");
      initialize_from_start(parameter, mesh, field, species);
  }
}

template<MixtureModel mix_model, TurbMethod turb_method>
void read_flowfield(cfd::Parameter &parameter, const cfd::Mesh &mesh, std::vector<Field> &field, Species &species) {
  const std::filesystem::path out_dir("output/field");
  if (!exists(out_dir)) {
    printf("The directory to flowfield files does not exist!\n");
  }
  MPI_File fp;
  MPI_File_open(MPI_COMM_WORLD, (out_dir.string() + "/flowfield.plt").c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &fp);
  MPI_Offset offset{0};
  MPI_Status status;

  // Magic number, 8 bytes + byte order + file type
  offset += 16;
  // "solution file"
  gxl::read_str_from_plt_MPI_ver(fp, offset);
  integer n_var_old{5};
  MPI_File_read_at(fp, offset, &n_var_old, 1, MPI_INT, &status);
  offset += 4;
  std::vector<std::string> var_name;
  var_name.resize(n_var_old);
  for (size_t i = 0; i < n_var_old; ++i) {
    var_name[i] = gxl::read_str_from_plt_MPI_ver(fp, offset);
  }
  // The first one tells if species info exists, if exists (1), else, (0).
  // The 2nd one tells if turbulent var exists, if 0 (compute from laminar), 1(From SA), 2(From SST)
  std::array old_data_info{0, 0};
  auto index_order = cfd::identify_variable_labels<mix_model, turb_method>(parameter, var_name, species,
                                                                           old_data_info);
  const integer n_spec{species.n_spec};
  const integer n_turb{parameter.get_int("n_turb")};

  integer *mx = new integer[mesh.n_block_total], *my = new integer[mesh.n_block_total], *mz = new integer[mesh.n_block_total];
  for (int b = 0; b < mesh.n_block_total; ++b) {
    // 1. Zone marker. Value = 299.0, indicates a V112 header.
    offset += 4;
    // 2. Zone name.
    gxl::read_str_from_plt_MPI_ver(fp, offset);
    // Jump through the following info which is not relevant to the current process.
    offset += 36;
    // For ordered zone, specify IMax, JMax, KMax
    MPI_File_read_at(fp, offset, &mx[b], 1, MPI_INT, &status);
    offset += 4;
    MPI_File_read_at(fp, offset, &my[b], 1, MPI_INT, &status);
    offset += 4;
    MPI_File_read_at(fp, offset, &mz[b], 1, MPI_INT, &status);
    offset += 4;
    // 11. For all zone types (repeat for each Auxiliary data name/value pair), no more data
    offset += 4;
  }
  // Read the EOHMARKER
  offset += 4;


  std::vector<std::string> zone_name;
  std::vector<double> solution_time;
  // Next, data section
  // Jump the front part for process 0 ~ myid-1
  integer n_jump_blk{0};
  for (int i = 0; i < parameter.get_int("myid"); ++i) {
    n_jump_blk += mesh.nblk[i];
  }
  integer i_blk{0};
  for (int b = 0; b < n_jump_blk; ++b) {
    offset += 16 + 20 * n_var_old;
    const integer N = mx[b] * my[b] * mz[b];
    // We always write double precision out
    offset += n_var_old * N * 8;
    ++i_blk;
  }
  // Read data of current process
  for (size_t blk = 0; blk < mesh.n_block; ++blk) {
    // 1. Zone marker. Value = 299.0, indicates a V112 header.
    offset += 4;
    // 2. Variable data format, 2 for double by default
    offset += 4 * n_var_old;
    // 3. Has passive variables: 0 = no, 1 = yes.
    offset += 4;
    // 4. Has variable sharing 0 = no, 1 = yes.
    offset += 4;
    // 5. Zero based zone number to share connectivity list with (-1 = no sharing).
    offset += 4;
    // 6. Compressed list of min/max pairs for each non-shared and non-passive variable.
    offset += 8 * 2 * n_var_old;
    // zone data
    // First, the coordinates x, y and z.
    const integer N = mx[i_blk] * my[i_blk] * mz[i_blk];
    offset += 3 * N * 8;
    // Other variables
    const auto &b = mesh[blk];
    MPI_Datatype ty;
    integer lsize[3]{mx[i_blk], my[i_blk], mz[i_blk]};
    const auto memsz = lsize[0] * lsize[1] * lsize[2] * 8;
    integer memsize[3]{b.mx + 2 * b.ngg, b.my + 2 * b.ngg, b.mz + 2 * b.ngg};
    const integer ngg_file{(mx[i_blk] - b.mx) / 2};
    integer start_idx[3]{b.ngg - ngg_file, b.ngg - ngg_file, b.ngg - ngg_file};
    MPI_Type_create_subarray(3, memsize, lsize, start_idx, MPI_ORDER_FORTRAN, MPI_DOUBLE, &ty);
    MPI_Type_commit(&ty);
    for (size_t l = 3; l < n_var_old; ++l) {
      auto index = index_order[l];
      if (index < 6) {
        // basic variables
        auto bv = field[blk].bv[index];
        MPI_File_read_at(fp, offset, bv, 1, ty, &status);
        offset += memsz;
      } else if (index < 6 + n_spec) {
        // If air, n_spec=0;
        // species variables
        index -= 6;
        auto sv = field[blk].sv[index];
        MPI_File_read_at(fp, offset, sv, 1, ty, &status);
        offset += memsz;
      } else if (index < 6 + n_spec + n_turb) {
        // If laminar, n_turb=0
        // turbulent variables
        index -= 6;
        if (n_turb == old_data_info[1]) {
          // SA from SA or SST from SST
          auto sv = field[blk].sv[index];
          MPI_File_read_at(fp, offset, sv, 1, ty, &status);
          offset += memsz;
        } else if (n_turb == 1 && old_data_info[1] == 2) {
          // SA from SST. Currently, just use freestream value to intialize. Modify this later when I write SA
          old_data_info[1] = 0;
        } else if (n_turb == 2 && old_data_info[1] == 1) {
          // SST from SA. As ACANS has done, the turbulent variables are intialized from freestream value
          old_data_info[1] = 0;
        }
      } else {
        // No matched label, just ignore
        offset += memsz;
      }
    }
    ++i_blk;
  }
  MPI_File_close(&fp);

  // Next, if the previous simulation does not contain some of the variables used in the current simulation,
  // then we initialize them here
  if constexpr (mix_model != MixtureModel::Air) {
    if (old_data_info[0] == 0) {
      initialize_spec_from_inflow(parameter, mesh, field, species);
    }
  }
  if constexpr (turb_method == TurbMethod::RANS) {
    if (old_data_info[1] == 0) {
      initialize_turb_from_inflow(parameter, mesh, field, species);
    }
  }

  std::ifstream step_file{"output/message/step.txt"};
  integer step{0};
  step_file >> step;
  step_file.close();
  parameter.update_parameter("step", step);

  if (parameter.get_int("myid") == 0) {
    printf("Flowfield is initialized from previous simulation results.\n");
  }
}

template<MixtureModel mix_model, TurbMethod turb_method>
std::vector<integer>
identify_variable_labels(cfd::Parameter &parameter, std::vector<std::string> &var_name, Species &species,
                         std::array<integer, 2> &old_data_info) {
  std::vector<integer> labels;
  const integer n_spec = species.n_spec;
  const integer n_turb = parameter.get_int("n_turb");
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
    } else {
      if constexpr (mix_model != MixtureModel::Air) {
        // We expect to find some species info. If not found, old_data_info[0] will remain 0.
        const auto &spec_name = species.spec_list;
        for (auto [spec, sp_label]: spec_name) {
          if (n == gxl::to_upper(spec)) {
            l = 6 + sp_label;
            old_data_info[0] = 1;
            break;
          }
        }
      }
      if constexpr (turb_method == TurbMethod::RANS) {
        // We expect to find some RANS variables. If not found, old_data_info[1] will remain 0.
        if (n == "K" || n == "TKE") { // turbulent kinetic energy
          if (n_turb == 2) {
            l = 6 + n_spec;
          }
          old_data_info[1] = 2; // SST model in previous simulation
        } else if (n == "OMEGA") { // specific dissipation rate
          if (n_turb == 2) {
            l = 6 + n_spec + 1;
          }
          old_data_info[1] = 2; // SST model in previous simulation
        } else if (n == "NUT SA") { // the variable from SA, not named yet!!!
          if (n_turb == 1) {
            l = 6 + n_spec;
          }
          old_data_info[1] = 1; // SA model in previous simulation
        }
      }
    }

    labels.emplace_back(l);
  }
  return labels;
}

}