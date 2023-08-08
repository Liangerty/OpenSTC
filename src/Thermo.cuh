#pragma once
#include "Define.h"

namespace cfd{
struct DParameter;
struct DZone;

__device__ void compute_enthalpy(real t, real *enthalpy, const DParameter* param);

__device__ void compute_cp(real t, real *cp, DParameter* param);

__device__ void compute_enthalpy_and_cp(real t, real *enthalpy, real *cp, const DParameter *param);

__device__ void compute_temperature(int i, int j, int k, const DParameter* param, DZone *zone);

__device__ void compute_gibbs_div_rt(real t, const DParameter* param, real* gibbs_rt);
}
