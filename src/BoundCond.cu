#include "BoundCond.cuh"
#include "Field.h"
#include "FieldOperation.cuh"
#include "SST.cuh"
#include "Parallel.h"

#ifdef GPU
namespace cfd {

template<MixtureModel mix_model, TurbMethod turb_method>
void cfd::Inflow<mix_model, turb_method>::copy_to_gpu(Inflow<mix_model, turb_method> *d_inflow, Species &spec,
                                                      const Parameter &parameter) {
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
void register_bc<Wall>(Wall *&walls, integer n_bc, std::vector<integer> &indices, BCInfo *&bc_info, Species &species,
                       Parameter &parameter) {
  if (n_bc <= 0) {
    return;
  }

  cudaMalloc(&walls, n_bc * sizeof(Wall));
  bc_info = new BCInfo[n_bc];
  for (integer i = 0; i < n_bc; ++i) {
    const integer index = indices[i];
    for (auto &bc_name: parameter.get_string_array("boundary_conditions")) {
      auto &bc = parameter.get_struct(bc_name);
      integer bc_label = std::get<integer>(bc.at("label"));
      if (index != bc_label) {
        continue;
      }
      Wall wall(bc);
      bc_info[i].label = bc_label;
      cudaMemcpy(&(walls[i]), &wall, sizeof(Wall), cudaMemcpyHostToDevice);
    }
  }
}

template<MixtureModel mix_model, TurbMethod turb_method>
void
register_inflow(Inflow<mix_model, turb_method> *&inflows, integer n_bc, std::vector<integer> &indices, BCInfo *&bc_info,
                Species &species,
                Parameter &parameter) {
  if (n_bc <= 0) {
    return;
  }

  cudaMalloc(&inflows, n_bc * sizeof(Inflow<mix_model, turb_method>));
  bc_info = new BCInfo[n_bc];
  for (integer i = 0; i < n_bc; ++i) {
    const integer index = indices[i];
    for (auto &bc_name: parameter.get_string_array("boundary_conditions")) {
      auto &bc = parameter.get_struct(bc_name);
      integer bc_label = std::get<integer>(bc.at("label"));
      if (index != bc_label) {
        continue;
      }
      bc_info[i].label = bc_label;
      Inflow<mix_model, turb_method> inflow(bc_name, species, parameter);
      inflow.copy_to_gpu(&(inflows[i]), species, parameter);
      break;
    }
  }
}

template<MixtureModel mix_model, TurbMethod turb_method>
void
DBoundCond<mix_model, turb_method>::apply_boundary_conditions(const Block &block, Field<mix_model, turb_method> &field,
                                                              DParameter *param) const {
  // Boundary conditions are applied in the order of priority, which with higher priority is applied later.
  // Finally, the communication between faces will be carried out after these bc applied
  // Priority: (-1 - inner faces >) 2-wall > 3-symmetry > 5-inflow > 6-outflow > 4-farfield

  // 4-farfield

  // 6-outflow
  for (size_t l = 0; l < n_outflow; l++) {
    const auto nb = outflow_info[l].n_boundary;
    for (size_t i = 0; i < nb; i++) {
      auto [i_zone, i_face] = outflow_info[l].boundary[i];
      if (i_zone != block.block_id) {
        continue;
      }
      const auto &h_f = block.boundary[i_face];
      const auto ngg = block.ngg;
      uint tpb[3], bpg[3];
      for (size_t j = 0; j < 3; j++) {
        auto n_point = h_f.range_end[j] - h_f.range_start[j] + 1;
        tpb[j] = n_point <= (2 * ngg + 1) ? 1 : 16;
        bpg[j] = (n_point - 1) / tpb[j] + 1;
      }
      dim3 TPB{tpb[0], tpb[1], tpb[2]}, BPG{bpg[0], bpg[1], bpg[2]};
      apply_outflow<turb_method> <<<BPG, TPB>>>(field.d_ptr, i_face);
    }
  }

  // 5-inflow
  for (size_t l = 0; l < n_inflow; l++) {
    const auto nb = inflow_info[l].n_boundary;
    for (size_t i = 0; i < nb; i++) {
      auto [i_zone, i_face] = inflow_info[l].boundary[i];
      if (i_zone != block.block_id) {
        continue;
      }
      const auto &hf = block.boundary[i_face];
      const auto ngg = block.ngg;
      uint tpb[3], bpg[3];
      for (size_t j = 0; j < 3; j++) {
        auto n_point = hf.range_end[j] - hf.range_start[j] + 1;
        tpb[j] = n_point <= (2 * ngg + 1) ? 1 : 16;
        bpg[j] = (n_point - 1) / tpb[j] + 1;
      }
      dim3 TPB{tpb[0], tpb[1], tpb[2]}, BPG{bpg[0], bpg[1], bpg[2]};
      apply_inflow<mix_model, turb_method> <<<BPG, TPB>>>(field.d_ptr, &inflow[l], param, i_face);
    }
  }

  // 3-symmetry
  for (size_t l = 0; l < n_symmetry; l++) {
    const auto nb = symmetry_info[l].n_boundary;
    for (size_t i = 0; i < nb; i++) {
      auto [i_zone, i_face] = symmetry_info[l].boundary[i];
      if (i_zone != block.block_id) {
        continue;
      }
      const auto &h_f = block.boundary[i_face];
      const auto ngg = block.ngg;
      uint tpb[3], bpg[3];
      for (size_t j = 0; j < 3; j++) {
        auto n_point = h_f.range_end[j] - h_f.range_start[j] + 1;
        tpb[j] = n_point <= (2 * ngg + 1) ? 1 : 16;
        bpg[j] = (n_point - 1) / tpb[j] + 1;
      }
      dim3 TPB{tpb[0], tpb[1], tpb[2]}, BPG{bpg[0], bpg[1], bpg[2]};
      apply_symmetry<mix_model, turb_method> <<<BPG, TPB>>>(field.d_ptr, i_face);
    }
  }

  // 2 - wall
  for (size_t l = 0; l < n_wall; l++) {
    const auto nb = wall_info[l].n_boundary;
    for (size_t i = 0; i < nb; i++) {
      auto [i_zone, i_face] = wall_info[l].boundary[i];
      if (i_zone != block.block_id) {
        continue;
      }
      const auto &hf = block.boundary[i_face];
      const auto ngg = block.ngg;
      uint tpb[3], bpg[3];
      for (size_t j = 0; j < 3; j++) {
        auto n_point = hf.range_end[j] - hf.range_start[j] + 1;
        tpb[j] = n_point <= (2 * ngg + 1) ? 1 : 16;
        bpg[j] = (n_point - 1) / tpb[j] + 1;
      }
      dim3 TPB{tpb[0], tpb[1], tpb[2]}, BPG{bpg[0], bpg[1], bpg[2]};
      apply_wall<mix_model, turb_method><<<BPG, TPB>>>(field.d_ptr, &wall[l], param, i_face);
    }
  }
}

template<MixtureModel mix_model, TurbMethod turb_method>
void
DBoundCond<mix_model, turb_method>::initialize_bc_on_GPU(Mesh &mesh, std::vector<Field<mix_model, turb_method>> &field,
                                                         Species &species, Parameter &parameter) {
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
  register_inflow(inflow, n_inflow, inflow_idx, inflow_info, species, parameter);
  register_bc<Outflow>(outflow, n_outflow, outflow_idx, outflow_info, species, parameter);

  link_bc_to_boundaries(mesh, field);

  MpiParallel::barrier();
  if (parameter.get_int("myid") == 0) {
    printf("Finish setting up boundary conditions.\n");
  }
}

template<MixtureModel mix_model, TurbMethod turb_method>
void DBoundCond<mix_model, turb_method>::link_bc_to_boundaries(Mesh &mesh,
                                                               std::vector<Field<mix_model, turb_method>> &field) const {
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

template
class DBoundCond<MixtureModel::Air, TurbMethod::Laminar>;

template
class DBoundCond<MixtureModel::Air, TurbMethod::RANS>;

template
class DBoundCond<MixtureModel::Mixture, TurbMethod::Laminar>;

template
class DBoundCond<MixtureModel::Mixture, TurbMethod::RANS>;

template
class DBoundCond<MixtureModel::FR, TurbMethod::Laminar>;

template
class DBoundCond<MixtureModel::FR, TurbMethod::RANS>;

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

template<MixtureModel mix_model, TurbMethod turb_method>
__global__ void apply_symmetry(DZone *zone, integer i_face) {
  const auto &b = zone->boundary[i_face];
  auto range_start = b.range_start, range_end = b.range_end;
  integer i = range_start[0] + (integer) (blockDim.x * blockIdx.x + threadIdx.x);
  integer j = range_start[1] + (integer) (blockDim.y * blockIdx.y + threadIdx.y);
  integer k = range_start[2] + (integer) (blockDim.z * blockIdx.z + threadIdx.z);
  if (i > range_end[0] || j > range_end[1] || k > range_end[2]) return;

  auto face = b.face;
  integer dir[]{0, 0, 0};
  dir[face] = b.direction;

  const integer inner_idx[3]{i - dir[0], j - dir[1], k - dir[2]};

  auto metric = zone->metric(i, j, k);
  real k_x{metric(face + 1, 1)}, k_y{metric(face + 1, 2)}, k_z{metric(face + 1, 3)};
  real k_magnitude = sqrt(k_x * k_x + k_y * k_y + k_z * k_z);
  k_x /= k_magnitude;
  k_y /= k_magnitude;
  k_z /= k_magnitude;

  auto &bv = zone->bv;
  real u1{bv(inner_idx[0], inner_idx[1], inner_idx[2], 1)}, v1{bv(inner_idx[0], inner_idx[1], inner_idx[2], 2)}, w1{
      bv(inner_idx[0], inner_idx[1], inner_idx[2], 3)};
  real u_k{k_x * u1 + k_y * v1 + k_z * w1};
  const real u_t{u1 - k_x * u_k}, v_t{v1 - k_y * u_k}, w_t{w1 - k_z * u_k};

  // The gradient of tangential velocity should be zero.
  bv(i, j, k, 1) = u_t;
  bv(i, j, k, 2) = v_t;
  bv(i, j, k, 3) = w_t;
  zone->vel(i, j, k) = std::sqrt(u_t * u_t + v_t * v_t + w_t * w_t);
  // The gradient of pressure, density, and scalars should also be zero.
  bv(i, j, k, 0) = bv(inner_idx[0], inner_idx[1], inner_idx[2], 0);
  bv(i, j, k, 4) = bv(inner_idx[0], inner_idx[1], inner_idx[2], 4);
  bv(i, j, k, 5) = bv(inner_idx[0], inner_idx[1], inner_idx[2], 5);
  auto &sv = zone->sv;
  for (integer l = 0; l < zone->n_scal; ++l) {
    sv(i, j, k, l) = sv(inner_idx[0], inner_idx[1], inner_idx[2], l);
  }

  // Assign values for conservative variables
  auto &cv = zone->cv;
  cv(i, j, k, 0) = bv(i, j, k, 0);
  cv(i, j, k, 1) = bv(i, j, k, 0) * bv(i, j, k, 1);
  cv(i, j, k, 2) = bv(i, j, k, 0) * bv(i, j, k, 2);
  cv(i, j, k, 3) = bv(i, j, k, 0) * bv(i, j, k, 3);
  // Scalars gradient are assumed to be 0, thus the difference in total energy is only caused by kinetic energy.
  cv(i, j, k, 4) = cv(inner_idx[0], inner_idx[1], inner_idx[2], 4) -
                   0.5 * bv(inner_idx[0], inner_idx[1], inner_idx[2], 0) * (u1 * u1 + v1 * v1 + w1 * w1) +
                   0.5 * bv(i, j, k, 0) * (u_t * u_t + v_t * v_t + w_t * w_t);
  for (integer l = 0; l < zone->n_scal; ++l) {
    cv(i, j, k, 5 + l) = bv(i, j, k, 0) * sv(i, j, k, l);
  }

  // For ghsot grids
  for (integer g = 1; g <= zone->ngg; ++g) {
    const integer gi{i + g * dir[0]}, gj{j + g * dir[1]}, gk{k + g * dir[2]};
    const integer ii{i - g * dir[0]}, ij{j - g * dir[1]}, ik{k - g * dir[2]};

    bv(gi, gj, gk, 0) = bv(ii, ij, ik, 0);

    auto &u{bv(ii, ij, ik, 1)}, v{bv(ii, ij, ik, 2)}, w{bv(ii, ij, ik, 3)};
    u_k = k_x * u + k_y * v + k_z * w;
    bv(gi, gj, gk, 1) = u - 2 * u_k * k_x;
    bv(gi, gj, gk, 2) = v - 2 * u_k * k_y;
    bv(gi, gj, gk, 3) = w - 2 * u_k * k_z;
    zone->vel(gi, gj, gk) = std::sqrt(bv(gi, gj, gk, 1) * bv(gi, gj, gk, 1) + bv(gi, gj, gk, 2) * bv(gi, gj, gk, 2) +
                                      bv(gi, gj, gk, 3) * bv(gi, gj, gk, 3));
    bv(gi, gj, gk, 4) = bv(ii, ij, ik, 4);
    bv(gi, gj, gk, 5) = bv(ii, ij, ik, 5);
    for (integer l = 0; l < zone->n_scal; ++l) {
      sv(gi, gj, gk, l) = sv(ii, ij, ik, l);
    }

    if constexpr (turb_method == TurbMethod::RANS) {
      zone->mut(gi, gj, gk) = zone->mut(ii, ij, ik);
    }
  }
}

template<TurbMethod turb_method>
__global__ void apply_outflow(DZone *zone, integer i_face) {
  const integer ngg = zone->ngg;
  integer dir[]{0, 0, 0};
  const auto &b = zone->boundary[i_face];
  dir[b.face] = b.direction;
  auto range_start = b.range_start, range_end = b.range_end;
  integer i = range_start[0] + (integer) (blockDim.x * blockIdx.x + threadIdx.x);
  integer j = range_start[1] + (integer) (blockDim.y * blockIdx.y + threadIdx.y);
  integer k = range_start[2] + (integer) (blockDim.z * blockIdx.z + threadIdx.z);
  if (i > range_end[0] || j > range_end[1] || k > range_end[2]) return;

  auto &bv = zone->bv;
  auto &sv = zone->sv;

  for (integer g = 1; g <= ngg; ++g) {
    const integer gi{i + g * dir[0]}, gj{j + g * dir[1]}, gk{k + g * dir[2]};
    for (integer l = 0; l < 6; ++l) {
      bv(gi, gj, gk, l) = bv(i, j, k, l);
    }
    for (integer l = 0; l < zone->n_scal; ++l) {
      sv(gi, gj, gk, l) = sv(i, j, k, l);
    }
    if constexpr (turb_method == TurbMethod::RANS) {
      zone->mut(gi, gj, gk) = zone->mut(i, j, k);
    }
  }
}

template<MixtureModel mix_model, TurbMethod turb_method>
__global__ void apply_inflow(DZone *zone, Inflow<mix_model, turb_method> *inflow, DParameter *param, integer i_face) {
  const integer ngg = zone->ngg;
  integer dir[]{0, 0, 0};
  const auto &b = zone->boundary[i_face];
  dir[b.face] = b.direction;
  auto range_start = b.range_start, range_end = b.range_end;
  integer i = range_start[0] + (integer) (blockDim.x * blockIdx.x + threadIdx.x);
  integer j = range_start[1] + (integer) (blockDim.y * blockIdx.y + threadIdx.y);
  integer k = range_start[2] + (integer) (blockDim.z * blockIdx.z + threadIdx.z);
  if (i > range_end[0] || j > range_end[1] || k > range_end[2]) return;

  auto &bv = zone->bv;
  auto &sv = zone->sv;

  const integer n_scalar = zone->n_scal;

  const real density = inflow->density;
  const real u = inflow->u;
  const real v = inflow->v;
  const real w = inflow->w;
  const auto *i_sv = inflow->sv;

  for (integer g = 1; g <= ngg; g++) {
    const integer gi{i + g * dir[0]}, gj{j + g * dir[1]}, gk{k + g * dir[2]};
    bv(gi, gj, gk, 0) = density;
    bv(gi, gj, gk, 1) = u;
    bv(gi, gj, gk, 2) = v;
    bv(gi, gj, gk, 3) = w;
    bv(gi, gj, gk, 4) = inflow->pressure;
    bv(gi, gj, gk, 5) = inflow->temperature;
    for (int l = 0; l < n_scalar; ++l) {
      sv(gi, gj, gk, l) = i_sv[l];
    }
    if constexpr (turb_method == TurbMethod::RANS) {
      zone->mut(gi, gj, gk) = inflow->mut;
    }
  }
}

template<MixtureModel mix_model, TurbMethod turb_method>
__global__ void apply_wall(DZone *zone, Wall *wall, DParameter *param, integer i_face) {
  const integer ngg = zone->ngg;
  integer dir[]{0, 0, 0};
  const auto &b = zone->boundary[i_face];
  dir[b.face] = b.direction;
  auto range_start = b.range_start, range_end = b.range_end;
  integer i = range_start[0] + (integer) (blockDim.x * blockIdx.x + threadIdx.x);
  integer j = range_start[1] + (integer) (blockDim.y * blockIdx.y + threadIdx.y);
  integer k = range_start[2] + (integer) (blockDim.z * blockIdx.z + threadIdx.z);
  if (i > range_end[0] || j > range_end[1] || k > range_end[2]) return;

  auto &bv = zone->bv;
  auto &cv = zone->cv;
  auto &sv = zone->sv;
  const integer n_spec = zone->n_spec;

  real t_wall{wall->temperature};

  const integer idx[]{i - dir[0], j - dir[1], k - dir[2]};
  if (wall->thermal_type == Wall::ThermalType::adiabatic) {
    t_wall = bv(idx[0], idx[1], idx[2], 5);
  }
  const real p{bv(idx[0], idx[1], idx[2], 4)};

  real mw{cfd::mw_air};
  if constexpr (mix_model != MixtureModel::Air) {
    // Mixture
    const auto mwk = param->mw;
    mw = 0;
    for (integer l = 0; l < n_spec; ++l) {
      sv(i, j, k, l) = sv(idx[0], idx[1], idx[2], l);
      mw += sv(i, j, k, l) / mwk[l];
    }
    mw = 1 / mw;
  }

  const real rho_wall = p * mw / (t_wall * cfd::R_u);
  bv(i, j, k, 0) = rho_wall;
  bv(i, j, k, 1) = 0;
  bv(i, j, k, 2) = 0;
  bv(i, j, k, 3) = 0;
  bv(i, j, k, 4) = p;
  bv(i, j, k, 5) = t_wall;
  cv(i, j, k, 0) = rho_wall;
  cv(i, j, k, 1) = 0;
  cv(i, j, k, 2) = 0;
  cv(i, j, k, 3) = 0;
  if constexpr (mix_model != MixtureModel::FL) {
    for (integer l = 0; l < n_spec; ++l) {
      cv(i, j, k, l + 5) = sv(i, j, k, l) * rho_wall;
    }
  }
  compute_total_energy<mix_model>(i, j, k, zone, param);

  // turbulent boundary condition
  if constexpr (turb_method == TurbMethod::RANS) {
    if (param->rans_model == 2) {
      // SST
      real mu_wall{0};
      if constexpr (mix_model != MixtureModel::Air) {
        mu_wall = compute_viscosity(i, j, k, t_wall, mw, param, zone);
      } else {
        mu_wall = Sutherland(t_wall);
      }
      const real dy = zone->wall_distance(idx[0], idx[1], idx[2]);
      sv(i, j, k, n_spec) = 0;
      cv(i, j, k, n_spec + 5) = 0;
      if (dy > 1e-25) {
        sv(i, j, k, n_spec + 1) = 60 * mu_wall / (rho_wall * SST::beta_1 * dy * dy);
      } else {
        sv(i, j, k, n_spec + 1) = sv(idx[0], idx[1], idx[2], n_spec + 1);
      }
      cv(i, j, k, n_spec + 6) = rho_wall * sv(i, j, k, n_spec + 1);
    }
  }

  for (int g = 1; g <= ngg; ++g) {
    const integer i_in[]{i - g * dir[0], j - g * dir[1], k - g * dir[2]};
    const integer i_gh[]{i + g * dir[0], j + g * dir[1], k + g * dir[2]};

    const real u_i{bv(i_in[0], i_in[1], i_in[2], 1)};
    const real v_i{bv(i_in[0], i_in[1], i_in[2], 2)};
    const real w_i{bv(i_in[0], i_in[1], i_in[2], 3)};
    const real p_i{bv(i_in[0], i_in[1], i_in[2], 4)};
    const real t_i{bv(i_in[0], i_in[1], i_in[2], 5)};

    double t_g{t_i};
    if (wall->thermal_type == Wall::ThermalType::isothermal) {
      t_g = 2 * t_wall - t_i;  // 0.5*(t_i+t_g)=t_w
      if (t_g <= 0.1 * t_wall) { // improve stability
        t_g = t_wall;
      }
    }

    if constexpr (mix_model != MixtureModel::Air) {
      const auto mwk = param->mw;
      mw = 0;
      for (integer l = 0; l < zone->n_spec; ++l) {
        sv(i_gh[0], i_gh[1], i_gh[2], l) = sv(i_in[0], i_in[1], i_in[2], l);
        mw += sv(i_gh[0], i_gh[1], i_gh[2], l) / mwk[l];
      }
      mw = 1 / mw;
    }

    const real rho_g{p_i * mw / (t_g * cfd::R_u)};
    bv(i_gh[0], i_gh[1], i_gh[2], 0) = rho_g;
    bv(i_gh[0], i_gh[1], i_gh[2], 1) = -u_i;
    bv(i_gh[0], i_gh[1], i_gh[2], 2) = -v_i;
    bv(i_gh[0], i_gh[1], i_gh[2], 3) = -w_i;
    bv(i_gh[0], i_gh[1], i_gh[2], 4) = p_i;
    bv(i_gh[0], i_gh[1], i_gh[2], 5) = t_g;

    // turbulent boundary condition
    if constexpr (turb_method == TurbMethod::RANS) {
      if (param->rans_model == 2) {
        // SST
        sv(i_gh[0], i_gh[1], i_gh[2], n_spec) = 0;
        sv(i_gh[0], i_gh[1], i_gh[2], n_spec + 1) = sv(i, j, k, n_spec + 1);
        zone->mut(i_gh[0], i_gh[1], i_gh[2]) = 0;
      }
    }
  }
}

} // cfd
#endif
