#pragma once
#include "Define.h"
#include "gxl_lib/Math.hpp"

namespace cfd{
// Limiter functions.
// 0-minmod
template<integer Method=0>
struct Limiter{
  __device__ real apply_limiter(real a, real b);
};

template<integer Method>
__device__ real Limiter<Method>::apply_limiter(real a, real b) {
  return 0.5 * (gxl::sgn(a) + gxl::sgn(b)) * min(std::abs(a), std::abs(b));
}


// https://stackoverflow.com/questions/25202250/c-template-instantiation-avoiding-long-switches
template<integer...> struct IntList{};

__device__
real apply_limiter(integer, IntList<>, real a, real b){}

template<integer I, integer...N>
__device__
real apply_limiter(integer i, IntList<I,N...>, real a, real b) {
  if (I != i)
    return apply_limiter(i, IntList<N...>(), a,b);

  Limiter<I> limiter;
  return limiter.apply_limiter(a,b);
}

template<integer ...N>
__device__
real apply_limiter(integer i, real a, real b) {
  return apply_limiter(i, IntList<N...>(), a, b);
}

template __device__ real apply_limiter<0, 1>(integer method, real a, real b);
}