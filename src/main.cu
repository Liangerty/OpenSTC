#include <cstdio>
#include "Parallel.h"
#include "Parameter.h"
#include "Mesh.h"
#include "Driver.cuh"

int main(int argc, char *argv[]) {
  cfd::MpiParallel mpi_parallel(&argc, &argv);

  cfd::Parameter parameter(mpi_parallel);

  cfd::Mesh mesh(parameter);

  bool species = parameter.get_bool("species");
  bool turbulent_laminar = parameter.get_bool("turbulence");
  bool reaction = parameter.get_bool("reaction");
  integer turbulent_method = parameter.get_int("turbulence_method");
  if (!turbulent_laminar) {
    turbulent_method = 0;
  }

  if (species) {
    // Multiple species
    if (turbulent_method == 1) {
      // RANS
      if (reaction) {
        // Finite rate chemistry
        cfd::Driver<MixtureModel::FR, TurbMethod::RANS> driver(parameter, mesh);
        driver.initialize_computation();
        driver.simulate();
      } else {
        // Pure mixing among species
        cfd::Driver<MixtureModel::Mixture, TurbMethod::RANS> driver(parameter, mesh);
        driver.initialize_computation();
        driver.simulate();
      }
    } else {
      // Laminar
      if (reaction) {
        // Finite rate chemistry
        cfd::Driver<MixtureModel::FR, TurbMethod::Laminar> driver(parameter, mesh);
        driver.initialize_computation();
        driver.simulate();
      } else {
        // Pure mixing among species
        cfd::Driver<MixtureModel::Mixture, TurbMethod::Laminar> driver(parameter, mesh);
        driver.initialize_computation();
        driver.simulate();
      }
    }
  } else {
    // Air simulation
    if (turbulent_method == 1) {
      // RANS and air
      cfd::Driver<MixtureModel::Air, TurbMethod::RANS> driver(parameter, mesh);
      driver.initialize_computation();
      driver.simulate();
    } else {
      // Laminar and air
      cfd::Driver<MixtureModel::Air, TurbMethod::Laminar> driver(parameter, mesh);
      driver.initialize_computation();
      driver.simulate();
    }
  }

  printf("Fuck off\n");
  return 0;
}
