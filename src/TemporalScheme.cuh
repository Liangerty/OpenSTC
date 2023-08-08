#pragma once
#include "Define.h"
#include "gxl_lib/Array.hpp"

namespace cfd {
struct DZone;
struct DParameter;
class Mesh;
class Parameter;

struct TemporalScheme {
  __device__ virtual void compute_time_step(DZone *zone, integer i, integer j, integer k, const DParameter *param) =0;
};

struct SteadyTemporalScheme:public TemporalScheme{
  __device__ void compute_time_step(DZone *zone, integer i, integer j, integer k, const DParameter *param) override;
};

struct LUSGS:public SteadyTemporalScheme{
//  __device__ void compute_time_step(DZone *zone, integer i, integer j, integer k, const DParameter *param) override;
};

} // cfd