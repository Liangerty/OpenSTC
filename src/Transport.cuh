#pragma once
#include "Define.h"
#include "ChemData.h"
#ifdef CUDACC
#include <cuda_runtime.h>
#endif
namespace cfd {
__host__ __device__
real Sutherland(real temperature);

real compute_viscosity(real temperature, real mw_total, real const *Y, Species &spec);

struct DParameter;
struct DZone;
__device__ void compute_transport_property(integer i, integer j, integer k, real temperature, real mw_total, const real *cp, DParameter* param, DZone* zone);

}