#include "Initialize.cuh"

namespace cfd {
void initialize_from_start(Parameter &parameter, const Mesh &mesh, std::vector<Field> &field, Species &species) {
  // First, find out how many groups of initial conditions are needed.
  const integer tot_group{parameter.get_int("groups_init")};
  std::vector<Inflow> groups_inflow;
  const std::string default_init = parameter.get_string("default_init");
  Inflow default_inflow(default_init, species, parameter);
  groups_inflow.push_back(default_inflow);

  std::vector<real> xs{}, xe{}, ys{}, ye{}, zs{}, ze{};
  if (tot_group > 1) {
    for (integer l = 0; l < tot_group - 1; ++l) {
      auto patch_struct_name = "init_cond_" + std::to_string(l);
      auto &patch_cond = parameter.get_struct(patch_struct_name);
      xs.push_back(std::get<real>(patch_cond.at("x0")));
      xe.push_back(std::get<real>(patch_cond.at("x1")));
      ys.push_back(std::get<real>(patch_cond.at("y0")));
      ye.push_back(std::get<real>(patch_cond.at("y1")));
      zs.push_back(std::get<real>(patch_cond.at("z0")));
      ze.push_back(std::get<real>(patch_cond.at("z1")));
      if (patch_cond.find("label") != patch_cond.cend()) {
        groups_inflow.emplace_back(std::get<std::string>(patch_cond.at("label")), species, parameter);
      } else {
        groups_inflow.emplace_back(patch_struct_name, species, parameter);
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

void initialize_spec_from_inflow(cfd::Parameter &parameter, const cfd::Mesh &mesh,
                                 std::vector<Field> &field, Species &species) {
  // This can also be implemented like the from_start one, which can have patch.
  // But currently, for easy to implement, just intialize the whole flowfield to the inflow composition,
  // which means that other species would have to be computed from boundary conditions.
  // If the need for initialize species in groups is strong,
  // then we implement it just by copying the previous function "initialize_from_start",
  // which should be easy.
  const std::string default_init = parameter.get_string("default_init");
  Inflow inflow(default_init, species, parameter);
  for (int blk = 0; blk < mesh.n_block; ++blk) {
    const integer mx{mesh[blk].mx}, my{mesh[blk].my}, mz{mesh[blk].mz};
    const auto n_spec = parameter.get_int("n_spec");
    auto mass_frac = inflow.sv;
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

void initialize_turb_from_inflow(cfd::Parameter &parameter, const cfd::Mesh &mesh,
                                 std::vector<Field> &field, Species &species) {
  // This can also be implemented like the from_start one, which can have patch.
  // But currently, for easy to implement, just intialize the whole flowfield to the main inflow turbulent state.
  // If the need for initialize turbulence in groups is strong,
  // then we implement it just by copying the previous function "initialize_from_start",
  // which should be easy.
  const std::string default_init = parameter.get_string("default_init");
  Inflow inflow(default_init, species, parameter);
  const auto n_turb = parameter.get_int("n_turb");
  const auto n_spec = parameter.get_int("n_spec");
  for (int blk = 0; blk < mesh.n_block; ++blk) {
    const integer mx{mesh[blk].mx}, my{mesh[blk].my}, mz{mesh[blk].mz};
    auto &sv = field[blk].sv;
    for (int k = 0; k < mz; ++k) {
      for (int j = 0; j < my; ++j) {
        for (int i = 0; i < mx; ++i) {
          for (integer l = 0; l < n_turb; ++l) {
            sv(i, j, k, n_spec + l) = inflow.sv[n_spec + l];
          }
        }
      }
    }
  }
  if (parameter.get_int("myid") == 0) {
    printf("Compute from laminar result. The turbulent field is initialized with freestream.\n");
  }
}

}
