#pragma once

#include "BoundCond.h"
#include "Mesh.h"

namespace cfd {

struct BCInfo {
  integer label = 0;
  integer n_boundary = 0;
  int2 *boundary = nullptr;
};

class Mesh;

struct DZone;
struct DParameter;
template<MixtureModel mix_model, TurbMethod turb_method>
struct Field;

template<MixtureModel mix_model, TurbMethod turb_method>
struct DBoundCond {
  DBoundCond() = default;

  void initialize_bc_on_GPU(Mesh &mesh, std::vector<Field<mix_model, turb_method>> &field, Species &species,
                            Parameter &parameter);

  void link_bc_to_boundaries(Mesh &mesh, std::vector<Field<mix_model, turb_method>> &field) const;

  void apply_boundary_conditions(const Block &block, Field<mix_model, turb_method> &field, DParameter *param) const;

  integer n_wall = 0, n_symmetry = 0, n_inflow = 0, n_outflow = 0;
  BCInfo *wall_info = nullptr;
  BCInfo *symmetry_info = nullptr;
  BCInfo *inflow_info = nullptr;
  BCInfo *outflow_info = nullptr;
  Wall *wall = nullptr;
  Symmetry *symmetry = nullptr;
  Inflow<mix_model, turb_method> *inflow = nullptr;
  Outflow *outflow = nullptr;
};

//void count_boundary_of_type_bc(const std::vector<Boundary>& boundary, integer n_bc, integer* sep, integer blk_idx,
//  integer n_block, BCInfo* bc_info);
void count_boundary_of_type_bc(const std::vector<Boundary>& boundary, integer n_bc, integer** sep, integer blk_idx,
  integer n_block, BCInfo* bc_info);

//void link_boundary_and_condition(const std::vector<Boundary>& boundary, BCInfo* bc, integer n_bc, const integer* sep,
//  integer i_zone);
void link_boundary_and_condition(const std::vector<Boundary>& boundary, BCInfo* bc, integer n_bc, integer** sep,
  integer i_zone);

template<TurbMethod turb_method>
__global__ void apply_outflow(DZone *zone, integer i_face);

template<MixtureModel mix_model, TurbMethod turb_method>
__global__ void apply_inflow(DZone *zone, Inflow<mix_model, turb_method> *inflow, DParameter *param, integer i_face);

template<MixtureModel mix_model, TurbMethod turb_method>
__global__ void apply_wall(DZone *zone, Wall *wall, DParameter *param, integer i_face);

template<MixtureModel mix_model, TurbMethod turb_method>
__global__ void apply_symmetry(DZone *zone, integer i_face);
} // cfd
