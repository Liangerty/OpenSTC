#include "Parallel.h"
#include <mpi.h>
#include <fstream>
#include <fmt/format.h>

cfd::MpiParallel::MpiParallel(int *argc, char ***argv) {
  MPI_Init(argc, argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_id);
  MPI_Comm_size(MPI_COMM_WORLD, &n_proc);

  if (my_id == 0) {
    std::ifstream par_file("input_files/setup/0_global_control.txt");
    std::string pass{};
    par_file >> pass >> pass >> pass >> parallel;
    par_file.close();
    if (parallel) {
      fmt::print("Parallel computation chosen! Number of processes: {}. \n", n_proc);
    } else {
      fmt::print("Serial computation chosen!\n");
    }
  }
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Bcast(&parallel, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
  if (my_id == 0) {
    if (!parallel && n_proc > 1) {
      fmt::print("You chose serial computation, but the number of processes is not equal to 1, n_proc={}.\n",
                 n_proc);
      MPI_Abort(MPI_COMM_WORLD, 1);
    }
    if (parallel && n_proc == 1) {
      fmt::print("You chose parallel computation, but the number of processes is equal to 1.\n");
      MPI_Abort(MPI_COMM_WORLD, 1);
    }
  }

  MPI_Barrier(MPI_COMM_WORLD);

  setup_gpu_device();
}

double cfd::MpiParallel::get_wall_time() { return MPI_Wtime(); }

void cfd::MpiParallel::barrier() { MPI_Barrier(MPI_COMM_WORLD); }

cfd::MpiParallel::~MpiParallel() { MPI_Finalize(); }

void cfd::MpiParallel::exit() { MPI_Abort(MPI_COMM_WORLD, 1); }
