#pragma once

#include "Parameter.h"
#include "Field.h"
#include <filesystem>
#include <fstream>

namespace cfd {
class Mesh;

template<MixtureModel mix_model, TurbMethod turb_method>
void wall_friction_heatflux_2d(const Mesh &mesh, const std::vector<cfd::Field<mix_model, turb_method>> &field,
                               const Parameter &parameter);

__global__ void wall_friction_heatFlux_2d(cfd::DZone *zone, real *friction, real *heat_flux, real dyn_pressure);

template<MixtureModel mix_model, TurbMethod turb_method>
void wall_friction_heatflux_2d(const Mesh &mesh, const std::vector<cfd::Field<mix_model, turb_method>> &field,
                               const Parameter &parameter) {
  const std::filesystem::path out_dir("output/wall");
  if (!exists(out_dir)) {
    create_directories(out_dir);
  }
  const auto path_name = out_dir.string();

  int size{mesh[0].mx};
  for (int blk = 1; blk < mesh.n_block; ++blk) {
    if (mesh[blk].mx > size) {
      size = mesh[blk].mx;
    }
  }
  std::vector<double> friction(size, 0);
  std::vector<double> heat_flux(size, 0);
  real *cf = nullptr, *qw = nullptr;
  cudaMalloc(&cf, size * sizeof(real));
  cudaMalloc(&qw, sizeof(real) * size);

  const double rho_inf = parameter.get_real("rho_inf");
  const double v_inf = parameter.get_real("v_inf");
  const double dyn_pressure = 0.5 * rho_inf * v_inf * v_inf;
  for (int blk = 0; blk < mesh.n_block; ++blk) {
    auto &block = mesh[blk];
    const int mx{block.mx};

    dim3 bpg((mx - 1) / 128 + 1, 1, 1);
    wall_friction_heatFlux_2d<<<bpg, 128>>>(field[blk].d_ptr, cf, qw, dyn_pressure);
    cudaMemcpy(friction.data(), cf, size * sizeof(real), cudaMemcpyDeviceToHost);
    cudaMemcpy(heat_flux.data(), qw, size * sizeof(real), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    std::ofstream f(path_name + "/friction_heatflux.dat");
    f << "variables = \"x\", \"cf\", \"qw\"\n";
    for (int i = 0; i < mx; ++i) {
      f << block.x(i, 0, 0) << '\t' << friction[i] << '\t' << heat_flux[i] << '\n';
    }
    f.close();
  }
  cudaFree(cf);
  cudaFree(qw);
}
}
