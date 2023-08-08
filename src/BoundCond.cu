#include "BoundCond.cuh"
#include "Parallel.h"

#ifdef GPU
namespace cfd {

template<typename BCType>
void register_bc(BCType *&bc, int n_bc, std::vector<integer> &indices, BCInfo *&bc_info, Species &species,
                 Parameter &parameter) {
  if (n_bc <= 0) {
    return;
  }
  cudaMalloc(&bc, n_bc * sizeof(BCType));
  bc_info = new BCInfo[n_bc];
  integer counter = 0;
  while (counter < n_bc) {
    BCType bctemp(indices[counter]);
    bc_info[counter].label = indices[counter];
    cudaMemcpy(&(bc[counter]), &bctemp, sizeof(BCType), cudaMemcpyHostToDevice);
    ++counter;
  }
}

template<>
void register_bc<Wall>(Wall *&bc, integer n_bc, std::vector<integer> &indices, BCInfo *&bc_info, Species &species,
                       Parameter &parameter) {
  if (n_bc <= 0) {
    return;
  }

  cudaMalloc(&bc, n_bc * sizeof(Wall));
  bc_info = new BCInfo[n_bc];
  for (integer i = 0; i < n_bc; ++i) {
    const integer index = indices[i];
    for (auto &bc_name: parameter.get_string_array("boundary_conditions")) {
      auto &this_bc = parameter.get_struct(bc_name);
      integer bc_label = std::get<integer>(this_bc.at("label"));
      if (index != bc_label) {
        continue;
      }
      Wall wall(this_bc);
      bc_info[i].label = bc_label;
      cudaMemcpy(&(bc[i]), &wall, sizeof(Wall), cudaMemcpyHostToDevice);
    }
  }
}

template<>
void register_bc<Inflow>(Inflow *&bc, integer n_bc, std::vector<integer> &indices, BCInfo *&bc_info, Species &species,
                         Parameter &parameter) {
  if (n_bc <= 0) {
    return;
  }

  cudaMalloc(&bc, n_bc * sizeof(Inflow));
  bc_info = new BCInfo[n_bc];
  for (integer i = 0; i < n_bc; ++i) {
    const integer index = indices[i];
    for (auto &bc_name: parameter.get_string_array("boundary_conditions")) {
      auto &this_bc = parameter.get_struct(bc_name);
      integer bc_label = std::get<integer>(this_bc.at("label"));
      if (index != bc_label) {
        continue;
      }
      bc_info[i].label = bc_label;
      Inflow inflow(bc_name, species, parameter);
      inflow.copy_to_gpu(&(bc[i]), species, parameter);
      break;
    }
  }
}

void DBoundCond::initialize_bc_on_GPU(Mesh &mesh, std::vector<Field> &field, Species &species, Parameter &parameter) {
  std::vector<integer> bc_labels;
  // Count the number of distinct boundary conditions
  for (auto i = 0; i < mesh.n_block; i++) {
    for (auto &b: mesh[i].boundary) {
      auto lab = b.type_label;
      bool has_this_bc = false;
      for (auto l: bc_labels) {
        if (l == lab) {
          has_this_bc = true;
          break;
        }
      }
      if (!has_this_bc) {
        bc_labels.push_back(lab);
      }
    }
  }
  // Initialize the inflow and wall condtions which are different among cases.
  std::vector<integer> wall_idx, symmetry_idx, inflow_idx, outflow_idx;
  auto &bcs = parameter.get_string_array("boundary_conditions");
  for (auto &bc_name: bcs) {
    auto &bc = parameter.get_struct(bc_name);
    auto label = std::get<integer>(bc.at("label"));

    auto this_iter = bc_labels.end();
    for (auto iter = bc_labels.begin(); iter != bc_labels.end(); ++iter) {
      if (*iter == label) {
        this_iter = iter;
        break;
      }
    }
    if (this_iter != bc_labels.end()) {
      bc_labels.erase(this_iter);
      auto type = std::get<std::string>(bc.at("type"));
      if (type == "wall") {
        wall_idx.push_back(label);
        ++n_wall;
      } else if (type == "inflow") {
        inflow_idx.push_back(label);
        ++n_inflow;
      }
        // Note: Normally, this would not happen for outflow, symmetry, and periodic conditions.
        // Because the above-mentioned conditions normally do not need to specify special treatments.
        // If we need to add supports for these conditions, then we add them here.
      else if (type == "outflow") {
        outflow_idx.push_back(label);
        ++n_outflow;
      } else if (type == "symmetry") {
        symmetry_idx.push_back(label);
        ++n_symmetry;
      }
    }
  }
  for (int lab: bc_labels) {
    if (lab == 2) {
      wall_idx.push_back(lab);
      ++n_wall;
    } else if (lab == 3) {
      symmetry_idx.push_back(lab);
      ++n_symmetry;
    } else if (lab == 5) {
      inflow_idx.push_back(lab);
      ++n_inflow;
    } else if (lab == 6) {
      outflow_idx.push_back(lab);
      ++n_outflow;
    }
  }

  // Read specific conditions
  register_bc<Wall>(wall, n_wall, wall_idx, wall_info, species, parameter);
  register_bc<Symmetry>(symmetry, n_symmetry, symmetry_idx, symmetry_info, species, parameter);
  register_bc<Inflow>(inflow, n_inflow, inflow_idx, inflow_info, species, parameter);
  register_bc<Outflow>(outflow, n_outflow, outflow_idx, outflow_info, species, parameter);

  link_bc_to_boundaries(mesh, field);

  MpiParallel::barrier();
  if (parameter.get_int("myid") == 0) {
    printf("Finish setting up boundary conditions.\n");
  }
}

void DBoundCond::link_bc_to_boundaries(Mesh &mesh, std::vector<Field> &field) const {
  const integer n_block{mesh.n_block};
  auto **i_wall = new integer *[n_wall];
  for (size_t i = 0; i < n_wall; i++) {
    i_wall[i] = new integer[n_block];
    for (integer j = 0; j < n_block; j++) {
      i_wall[i][j] = 0;
    }
  }
  auto **i_symm = new integer *[n_symmetry];
  for (size_t i = 0; i < n_symmetry; i++) {
    i_symm[i] = new integer[n_block];
    for (integer j = 0; j < n_block; j++) {
      i_symm[i][j] = 0;
    }
  }
  auto **i_inflow = new integer *[n_inflow];
  for (size_t i = 0; i < n_inflow; i++) {
    i_inflow[i] = new integer[n_block];
    for (integer j = 0; j < n_block; j++) {
      i_inflow[i][j] = 0;
    }
  }
  auto **i_outflow = new integer *[n_outflow];
  for (size_t i = 0; i < n_outflow; i++) {
    i_outflow[i] = new integer[n_block];
    for (integer j = 0; j < n_block; j++) {
      i_outflow[i][j] = 0;
    }
  }

  // We first count how many faces corresponds to a given boundary condition
  for (integer i = 0; i < n_block; i++) {
    count_boundary_of_type_bc(mesh[i].boundary, n_wall, i_wall, i, n_block, wall_info);
    count_boundary_of_type_bc(mesh[i].boundary, n_symmetry, i_symm, i, n_block, symmetry_info);
    count_boundary_of_type_bc(mesh[i].boundary, n_inflow, i_inflow, i, n_block, inflow_info);
    count_boundary_of_type_bc(mesh[i].boundary, n_outflow, i_outflow, i, n_block, outflow_info);
  }
  for (size_t l = 0; l < n_wall; l++) {
    wall_info[l].boundary = new int2[wall_info[l].n_boundary];
  }
  for (size_t l = 0; l < n_symmetry; ++l) {
    symmetry_info[l].boundary = new int2[symmetry_info[l].n_boundary];
  }
  for (size_t l = 0; l < n_inflow; l++) {
    inflow_info[l].boundary = new int2[inflow_info[l].n_boundary];
  }
  for (size_t l = 0; l < n_outflow; l++) {
    outflow_info[l].boundary = new int2[outflow_info[l].n_boundary];
  }

  const auto ngg{mesh[0].ngg};
  for (auto i = 0; i < n_block; i++) {
    link_boundary_and_condition(mesh[i].boundary, wall_info, n_wall, i_wall, i);
    link_boundary_and_condition(mesh[i].boundary, symmetry_info, n_symmetry, i_symm, i);
    link_boundary_and_condition(mesh[i].boundary, inflow_info, n_inflow, i_inflow, i);
    link_boundary_and_condition(mesh[i].boundary, outflow_info, n_outflow, i_outflow, i);
  }
  for (auto i = 0; i < n_block; i++) {
    for (size_t l = 0; l < n_wall; l++) {
      const auto nb = wall_info[l].n_boundary;
      for (size_t m = 0; m < nb; m++) {
        auto i_zone = wall_info[l].boundary[m].x;
        if (i_zone != i) {
          continue;
        }
        auto &b = mesh[i].boundary[wall_info[l].boundary[m].y];
        for (int q = 0; q < 3; ++q) {
          if (q == b.face) continue;
          b.range_start[q] += ngg;
          b.range_end[q] -= ngg;
        }
      }
    }
    cudaMemcpy(field[i].h_ptr->boundary, mesh[i].boundary.data(), mesh[i].boundary.size() * sizeof(Boundary),
               cudaMemcpyHostToDevice);
  }
  for (integer i = 0; i < n_wall; i++) {
    delete[]i_wall[i];
  }
  for (integer i = 0; i < n_symmetry; i++) {
    delete[]i_symm[i];
  }
  for (integer i = 0; i < n_inflow; i++) {
    delete[]i_inflow[i];
  }
  for (integer i = 0; i < n_outflow; i++) {
    delete[]i_outflow[i];
  }
  delete[]i_wall;
  delete[]i_symm;
  delete[]i_inflow;
  delete[]i_outflow;
}

void count_boundary_of_type_bc(const std::vector<Boundary> &boundary, integer n_bc, integer **sep, integer blk_idx,
                               integer n_block, BCInfo *bc_info) {
  if (n_bc <= 0) {
    return;
  }

  // Count how many faces correspond to the given bc
  const auto n_boundary{boundary.size()};
  auto *n = new integer[n_bc];
  memset(n, 0, sizeof(integer) * n_bc);
  for (size_t l = 0; l < n_bc; l++) {
    integer label = bc_info[l].label; // This means every bc should have a member "label"
    for (size_t i = 0; i < n_boundary; i++) {
      auto &b = boundary[i];
      if (b.type_label == label) {
        ++bc_info[l].n_boundary;
        ++n[l];
      }
    }
  }
  if (blk_idx < n_block - 1) {
    for (size_t l = 0; l < n_bc; l++) {
      sep[l][blk_idx + 1] = n[l] + sep[l][blk_idx];
    }
  }
  delete[]n;
}

void link_boundary_and_condition(const std::vector<Boundary> &boundary, BCInfo *bc, integer n_bc, integer **sep,
                                 integer i_zone) {
  const auto n_boundary{boundary.size()};
  for (size_t l = 0; l < n_bc; l++) {
    integer label = bc[l].label;
    int has_read{sep[l][i_zone]};
    for (auto i = 0; i < n_boundary; i++) {
      auto &b = boundary[i];
      if (b.type_label == label) {
        bc[l].boundary[has_read] = make_int2(i_zone, i);
        ++has_read;
      }
    }
  }
}

void Inflow::copy_to_gpu(Inflow *d_inflow, Species &spec, const Parameter &parameter) {
  const integer n_scalar{parameter.get_int("n_scalar")};
  real *h_sv = new real[n_scalar];
  for (integer l = 0; l < n_scalar; ++l) {
    h_sv[l] = sv[l];
  }
  delete[]sv;
  cudaMalloc(&sv, n_scalar * sizeof(real));
  cudaMemcpy(sv, h_sv, n_scalar * sizeof(real), cudaMemcpyHostToDevice);

  cudaMemcpy(d_inflow, this, sizeof(Inflow), cudaMemcpyHostToDevice);
}
} // cfd
#endif
