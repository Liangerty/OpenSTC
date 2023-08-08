#pragma once

#include "Define.h"
#include "gxl_lib/Array.hpp"
#include "Parameter.h"
#include "Mesh.h"

namespace cfd {
template<MixtureModel mix_model, TurbMethod turb_method>
struct Inflow;

#ifdef GPU

struct DZone {
  DZone() = default;

  integer mx = 0, my = 0, mz = 0, ngg = 0, n_spec = 0, n_scal = 0, n_var = 5;
  ggxl::Array3D<real> x, y, z;
  Boundary *boundary= nullptr;
  InnerFace *innerface= nullptr;
  ParallelFace *parface= nullptr;
  ggxl::Array3D<real> jac;
  ggxl::Array3D<gxl::Matrix<real, 3, 3, 1>> metric;
  ggxl::VectorField3D<real> cv; // Conservative variable: 0-:rho, 1-:rho*u, 2-:rho*v, 3-:rho*w, 4-:rho*(E+V*V/2), 5->(4+Ns)-:rho*Y
  ggxl::VectorField3D<real> bv; // Basic variable: 0-density, 1-u, 2-v, 3-w, 4-pressure, 5-temperature
  ggxl::VectorField3D<real> sv; // Scalar variables: [0,n_spec) - mass fractions; [n_spec,n_spec+n_turb) - turbulent variables
  ggxl::VectorField3D<real> bv_last; // Basic variable of last step: 0-density, 1-velocity magnitude, 2-pressure, 3-temperature
  ggxl::Array3D<real> vel;      // Velocity magnitude
  ggxl::Array3D<real> acoustic_speed;
  ggxl::Array3D<real> mach;     // Mach number
  ggxl::Array3D<real> mul;      // Dynamic viscosity
  ggxl::Array3D<real> conductivity;      // Dynamic viscosity

  // Mixture variables
  ggxl::VectorField3D<real> rho_D; // the mass diffusivity of species
  ggxl::Array3D<real> gamma;  // specific heat ratio
  // Turbulent variables
  ggxl::Array3D<real> mut;  // turbulent viscosity
  // Flamelet variables
  ggxl::Array3D<real> scalar_diss_rate;  // scalar dissipation rate

  // Variables used in computation
  ggxl::VectorField3D<real> dq; // The residual for flux computing
  ggxl::Array3D<real[3]> inv_spectr_rad;  // inviscid spectral radius. Used when LUSGS type temporal scheme is used.
  ggxl::Array3D<real> visc_spectr_rad;  // viscous spectral radius.
  ggxl::Array3D<real> dt_local; //local time step. Used for steady flow simulation
};

#endif

template<MixtureModel mix_model, TurbMethod turb_method>
struct Field{
  Field(Parameter &parameter, const Block &block_in);

  void initialize_basic_variables(const Parameter &parameter, const std::vector<Inflow<mix_model,turb_method>> &inflows,
                                  const std::vector<real> &xs, const std::vector<real> &xe, const std::vector<real> &ys,
                                  const std::vector<real> &ye, const std::vector<real> &zs,
                                  const std::vector<real> &ze);

  void setup_device_memory(const Parameter &parameter);

  void copy_data_from_device();
  
  const Block& block;
  gxl::VectorField3D<real> cv;  // Is it used in data communication? If not, this can be deleted, because all computations including cv are executed on GPU
  gxl::VectorField3D<real> bv;  // basic variables, including density, u, v, w, p, temperature
  gxl::VectorField3D<real> sv;  // passive scalar variables, including species mass fractions, turbulent variables, mixture fractions, etc.
  gxl::VectorField3D<real> ov;  // other variables used in the computation, e.g., the Mach number, the mut in turbulent computation, scalar dissipation rate in flamelet, etc.
#ifdef GPU
  DZone *d_ptr = nullptr;
  DZone *h_ptr = nullptr;
#endif
};
}