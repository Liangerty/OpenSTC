#include "PostProcess.h"
#include "Field.h"
#include <filesystem>
#include <fstream>

void cfd::wall_friction_heatflux_2d(const Mesh &mesh, const std::vector<cfd::Field> &field, const Parameter &parameter) {
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

__global__ void cfd::wall_friction_heatFlux_2d(cfd::DZone *zone, real *friction, real *heat_flux, real dyn_pressure) {
  const integer i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i>=zone->mx) return;

  auto &pv = zone->bv;

  auto &metric = zone->metric(i, 0, 0);
  const real xi_x = metric(1, 1), xi_y = metric(1, 2);
  const real eta_x = metric(2, 1), eta_y = metric(2, 2);

  const real viscosity = zone->mul(i, 0, 0);
  const double u_parallel_wall = (xi_x * pv(i, 1, 0, 1) + xi_y * pv(i, 1, 0, 2)) / sqrt(xi_x * xi_x + xi_y * xi_y);
  const double grad_eta = sqrt(eta_x * eta_x + eta_y * eta_y);
  const double du_normal_wall = u_parallel_wall * grad_eta;
  // dimensionless fiction coefficient, cf
  friction[i] = viscosity * du_normal_wall / dyn_pressure;

  const double conductivity = zone->thermal_conductivity(i, 0, 0);
  // dimensional heat flux
  heat_flux[i] = conductivity * (pv(i, 1, 0, 5) - pv(i, 0, 0, 5)) * grad_eta;

}
