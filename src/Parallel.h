#pragma once

namespace cfd {
/**
 * \brief The class controlling the MPI message of current simulation
 */
class MpiParallel {
public:
  /** \brief number of processes in current simulation */
  int n_proc{1};
  /** \brief The id of current process in all processes */
  int my_id{0};
  /** \brief if the current simulation is a parallel simulation */
  inline static bool parallel{false};

  MpiParallel(int *argc, char ***argv);

  MpiParallel() = delete;

  MpiParallel(const MpiParallel &) = delete;

  MpiParallel(MpiParallel &&) = delete;

  MpiParallel &operator=(const MpiParallel &) = delete;

  MpiParallel operator=(MpiParallel &&) = delete;


  static double get_wall_time();

  static void barrier();

  static void exit();

  ~MpiParallel();

private:
  void setup_gpu_device() const;
};
}
