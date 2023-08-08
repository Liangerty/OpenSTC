#pragma once

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER
#define __host__
#endif

using integer = int;
using real = double;
using uint = unsigned int;

enum class TurbMethod{
  Laminar,
  RANS,
  LES,
//  ILES,
//  DNS
};

enum class MixtureModel{
  Air,
  Mixture,  // Species mixing
  FR,       // Finite Rate
  FL,       // Flamelet Model
};

enum class CombModel{
  NoReaction,
  FiniteRate,
  Flamelet
};

template<MixtureModel T>
constexpr bool is_mixture = true;

template<>
constexpr bool is_mixture<MixtureModel::Air> = false;
