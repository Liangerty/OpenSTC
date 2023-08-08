#include "PostProcess.h"

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
