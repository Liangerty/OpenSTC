#include "Driver.cuh"
#include "Initialize.cuh"
#include "DataCommunication.cuh"
#include "TimeAdvanceFunc.cuh"
#include "SourceTerm.cuh"
#include "SchemeSelector.cuh"
#include "ImplicitTreatmentHPP.cuh"
#include "Parallel.h"
#include "PostProcess.h"
#include "MPIIO.hpp"

namespace cfd {

template<MixtureModel mix_model, TurbMethod turb_method>
Driver<mix_model, turb_method>::Driver(Parameter &parameter, Mesh &mesh_):
    myid(parameter.get_int("myid")), time(), mesh(mesh_), parameter(parameter),
    spec(parameter), reac(parameter, spec) {
  // Allocate the memory for every block
  for (integer blk = 0; blk < mesh.n_block; ++blk) {
    field.emplace_back(parameter, mesh[blk]);
  }

  initialize_basic_variables<mix_model, turb_method>(parameter, mesh, field, spec);

  if (parameter.get_int("initial") == 1) {
    // If continue from previous results, then we need the residual scales
    // If the file does not exist, then we have a trouble
    std::ifstream res_scale_in("output/message/residual_scale.txt");
    res_scale_in >> res_scale[0] >> res_scale[1] >> res_scale[2] >> res_scale[3];
    res_scale_in.close();
  }
#ifdef GPU
  for (integer blk = 0; blk < mesh.n_block; ++blk) {
    field[blk].setup_device_memory(parameter);
  }
  bound_cond.initialize_bc_on_GPU(mesh_, field, spec, parameter);
  DParameter d_param(parameter, spec, reac);
  cudaMalloc(&param, sizeof(DParameter));
  cudaMemcpy(param, &d_param, sizeof(DParameter), cudaMemcpyHostToDevice);
#endif
}

template<MixtureModel mix_model, TurbMethod turb_method>
void Driver<mix_model, turb_method>::initialize_computation() {
  dim3 tpb{8, 8, 4};
  if (mesh.dimension == 2) {
    tpb = {16, 16, 1};
  }
  const auto ng_1 = 2 * mesh[0].ngg - 1;

  // If we use k-omega SST model, we need the wall distance, thus we need to compute or read it here.
  if constexpr (turb_method == TurbMethod::RANS) {
    if (parameter.get_int("RANS_model") == 2) {
      // SST method
      acquire_wall_distance();
    }
  }

  if (mesh.dimension == 2) {
    for (auto b = 0; b < mesh.n_block; ++b) {
      const auto mx{mesh[b].mx}, my{mesh[b].my};
      dim3 BPG{(mx + ng_1) / tpb.x + 1, (my + ng_1) / tpb.y + 1, 1};
      eliminate_k_gradient <<<BPG, tpb >>>(field[b].d_ptr);
    }
  }

  // Second, apply boundary conditions to all boundaries, including face communication between faces
  for (integer b = 0; b < mesh.n_block; ++b) {
    bound_cond.apply_boundary_conditions<mix_model, turb_method>(mesh[b], field[b], param);
  }
  if (myid == 0) {
    printf("Boundary conditions are applied successfully for initialization\n");
  }


  // First, compute the conservative variables from basic variables
  for (auto i = 0; i < mesh.n_block; ++i) {
    integer mx{mesh[i].mx}, my{mesh[i].my}, mz{mesh[i].mz};
    dim3 bpg{(mx + ng_1) / tpb.x + 1, (my + ng_1) / tpb.y + 1, (mz + ng_1) / tpb.z + 1};
    compute_cv_from_bv<mix_model, turb_method><<<bpg, tpb>>>(field[i].d_ptr, param);
    if constexpr (turb_method == TurbMethod::RANS) {
      // We need the wall distance here. And the mut are computed for bc
      initialize_mut<mix_model><<<bpg, tpb >>>(field[i].d_ptr, param);
    }
  }
  cudaDeviceSynchronize();
  // Third, communicate values between processes
  data_communication<mix_model, turb_method>(mesh, field, parameter, 0, param);

  if (myid == 0) {
    printf("Finish data transfer.\n");
  }
  cudaDeviceSynchronize();

  for (auto b = 0; b < mesh.n_block; ++b) {
    integer mx{mesh[b].mx}, my{mesh[b].my}, mz{mesh[b].mz};
    dim3 bpg{(mx + ng_1) / tpb.x + 1, (my + ng_1) / tpb.y + 1, (mz + ng_1) / tpb.z + 1};
    update_physical_properties<mix_model, turb_method><<<bpg, tpb>>>(field[b].d_ptr, param);
  }
  cudaDeviceSynchronize();
  if (myid == 0) {
    printf("The flowfield is completely initialized on GPU.\n");
  }
}

template<MixtureModel mix_model, TurbMethod turb_method>
void Driver<mix_model, turb_method>::simulate() {
  const auto steady{parameter.get_bool("steady")};
  if (steady) {
    steady_simulation();
  } else {
    const auto temporal_tag{parameter.get_int("temporal_scheme")};
    switch (temporal_tag) {
      case 11: // For example, if DULUSGS, then add a function to initiate the computation instead of initialize before setting up the scheme as CPU code
        break;
      case 12:break;
      default:printf("Not implemented");
    }
  }
}

template<MixtureModel mix_model, TurbMethod turb_method>
void Driver<mix_model, turb_method>::acquire_wall_distance() {
  integer method_for_wall_distance{parameter.get_int("wall_distance")};
  if (method_for_wall_distance == 0) {
    // We need to compute it.

    // Store all wall coordinates of this process in the vector
    std::vector<real> wall_coor;
    for (integer iw = 0; iw < bound_cond.n_wall; ++iw) {
      auto &info = bound_cond.wall_info[iw];
      const auto nb = info.n_boundary;
      for (size_t m = 0; m < nb; m++) {
        auto i_zone = info.boundary[m].x;
        auto &x = mesh[i_zone].x;
        auto &y = mesh[i_zone].y;
        auto &z = mesh[i_zone].z;
        auto &f = mesh[i_zone].boundary[info.boundary[m].y];
        for (integer k = f.range_start[2]; k <= f.range_end[2]; ++k) {
          for (integer j = f.range_start[1]; j <= f.range_end[1]; ++j) {
            for (integer i = f.range_start[0]; i <= f.range_end[0]; ++i) {
              wall_coor.push_back(x(i, j, k));
              wall_coor.push_back(y(i, j, k));
              wall_coor.push_back(z(i, j, k));
            }
          }
        }
      }
    }
    const integer n_proc{parameter.get_int("n_proc")};
    auto *n_wall_point = new integer[n_proc];
    auto n_wall_this = static_cast<integer>(wall_coor.size());
    MPI_Allgather(&n_wall_this, 1, MPI_INT, n_wall_point, 1, MPI_INT, MPI_COMM_WORLD);
    auto *disp = new integer[n_proc];
    disp[0] = 0;
    for (integer i = 1; i < n_proc; ++i) {
      disp[i] = disp[i - 1] + n_wall_point[i - 1];
    }
    integer total_wall_number{0};
    for (integer i = 0; i < n_proc; ++i) {
      total_wall_number += n_wall_point[i];
    }
    std::vector<real> wall_points(total_wall_number, 0);
    // NOTE: The MPI process here is not examined carefully, if there are mistakes or things hard to understand, examine here.
    MPI_Allgatherv(wall_coor.data(), n_wall_point[myid], MPI_DOUBLE, wall_points.data(), n_wall_point, disp, MPI_DOUBLE,
                   MPI_COMM_WORLD);
    real *wall_corr_gpu = nullptr;
    cudaMalloc(&wall_corr_gpu, total_wall_number * sizeof(real));
    cudaMemcpy(wall_corr_gpu, wall_points.data(), total_wall_number * sizeof(real), cudaMemcpyHostToDevice);
    if (myid == 0) {
      printf("Start computing wall distance.\n");
    }
    for (integer blk = 0; blk < mesh.n_block; ++blk) {
      const integer ngg{mesh[0].ngg};
      const integer mx{mesh[blk].mx + 2 * ngg}, my{mesh[blk].my + 2 * ngg}, mz{mesh[blk].mz + 2 * ngg};
      dim3 tpb{512, 1, 1};
      dim3 bpg{(mx - 1) / tpb.x + 1, (my - 1) / tpb.y + 1, (mz - 1) / tpb.z + 1};
      compute_wall_distance<<<bpg, tpb>>>(wall_corr_gpu, field[blk].d_ptr, total_wall_number);
//      cudaMemcpy(field[blk].var_without_ghost_grid.data(), field[blk].h_ptr->wall_distance.data(), field[blk].h_ptr->wall_distance.size()*sizeof(real),cudaMemcpyDeviceToHost);
    }
    cudaDeviceSynchronize();
    if (myid == 0) {
      printf("Finish computing wall distance.\n");
    }

  } else {
    // We need to read it.
  }
}

template<MixtureModel mix_model, TurbMethod turb_method>
void Driver<mix_model, turb_method>::steady_simulation() {
  if (myid == 0) {
    printf("Steady flow simulation.\n");
  }
  bool converged{false};
  integer step{parameter.get_int("step")};
  integer total_step{parameter.get_int("total_step") + step};
  const integer n_block{mesh.n_block};
  const integer n_var{parameter.get_int("n_var")};
  const integer ngg{mesh[0].ngg};
  const integer ng_1 = 2 * ngg - 1;
  const integer output_screen = parameter.get_int("output_screen");
  const integer output_file = parameter.get_int("output_file");

  MPIIO<mix_model, turb_method> mpiio(myid, mesh, field, parameter, spec, 0);

  dim3 tpb{8, 8, 4};
  if (mesh.dimension == 2) {
    tpb = {16, 16, 1};
  }
  dim3 *bpg = new dim3[n_block];
  for (integer b = 0; b < n_block; ++b) {
    const auto mx{mesh[b].mx}, my{mesh[b].my}, mz{mesh[b].mz};
    bpg[b] = {(mx - 1) / tpb.x + 1, (my - 1) / tpb.y + 1, (mz - 1) / tpb.z + 1};
  }

  while (!converged) {
    ++step;
    /*[[unlikely]]*/if (step > total_step) {
      break;
    }

    // Start a single iteration
    // First, store the value of last step
    if (step % output_screen == 0) {
      for (auto b = 0; b < n_block; ++b) {
        store_last_step<<<bpg[b], tpb>>>(field[b].d_ptr);
      }
    }

    for (auto b = 0; b < n_block; ++b) {
      // Set dq to 0
      cudaMemset(field[b].h_ptr->dq.data(), 0, field[b].h_ptr->dq.size() * n_var * sizeof(real));

      // Second, for each block, compute the residual dq
      // First, compute the source term, because properties such as mut are updated here.
      compute_source<mix_model, turb_method><<<bpg[b], tpb>>>(field[b].d_ptr, param);
      compute_inviscid_flux<mix_model, turb_method>(mesh[b], field[b].d_ptr, param, n_var);
      compute_viscous_flux<mix_model, turb_method>(mesh[b], field[b].d_ptr, param, n_var);

      // compute local time step
      local_time_step<mix_model, turb_method><<<bpg[b], tpb>>>(field[b].d_ptr, param);
      // implicit treatment if needed
      implicit_treatment<mix_model, turb_method>(mesh[b], param, field[b].d_ptr, parameter, field[b].h_ptr);

      // update conservative and basic variables
      update_cv_and_bv<mix_model, turb_method><<<bpg[b], tpb>>>(field[b].d_ptr, param);

      // limit unphysical values computed by the program
      limit_flow<<<bpg[b], tpb>>>(field[b].d_ptr, param, b);

      // apply boundary conditions
      bound_cond.apply_boundary_conditions<mix_model, turb_method>(mesh[b], field[b], param);
    }
    // Third, transfer data between and within processes
    data_communication<mix_model, turb_method>(mesh, field, parameter, step, param);

    if (mesh.dimension == 2) {
      for (auto b = 0; b < n_block; ++b) {
        const auto mx{mesh[b].mx}, my{mesh[b].my};
        dim3 BPG{(mx + ng_1) / tpb.x + 1, (my + ng_1) / tpb.y + 1, 1};
        eliminate_k_gradient<<<BPG, tpb>>>(field[b].d_ptr);
      }
    }

    // update physical properties such as Mach number, transport coefficients et, al.
    for (auto b = 0; b < n_block; ++b) {
      integer mx{mesh[b].mx}, my{mesh[b].my}, mz{mesh[b].mz};
      dim3 BPG{(mx + 1) / tpb.x + 1, (my + 1) / tpb.y + 1, (mz + 1) / tpb.z + 1};
      update_physical_properties<mix_model, turb_method><<<BPG, tpb>>>(field[b].d_ptr, param);
    }

    // Finally, test if the simulation reaches convergence state
    if (step % output_screen == 0) {
      real err_max = compute_residual(step);
      converged = err_max < parameter.get_real("convergence_criteria");
      if (myid == 0) {
        steady_screen_output(step, err_max);
      }
    }
    cudaDeviceSynchronize();
    if (step % output_file == 0 || converged) {
      mpiio.print_field(step);
      post_process();
    }
  }
  delete[] bpg;
}

template<MixtureModel mix_model, TurbMethod turb_method>
real Driver<mix_model, turb_method>::compute_residual(integer step) {
  const integer n_block{mesh.n_block};
  for (auto &e: res) {
    e = 0;
  }

  dim3 tpb{8, 8, 4};
  if (mesh.dimension == 2) {
    tpb = {16, 16, 1};
  }
  for (integer b = 0; b < n_block; ++b) {
    const auto mx{mesh[b].mx}, my{mesh[b].my}, mz{mesh[b].mz};
    dim3 bpg = {(mx - 1) / tpb.x + 1, (my - 1) / tpb.y + 1, (mz - 1) / tpb.z + 1};
    // compute the square of the difference of the basic variables
    compute_square_of_dbv<<<bpg, tpb>>>(field[b].d_ptr);
  }

  constexpr integer TPB{128};
  constexpr integer n_res_var{4};
  real res_block[n_res_var];
  int num_sms, num_blocks_per_sm;
  cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0);
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks_per_sm, reduction_of_dv_squared<n_res_var>, TPB,
                                                TPB * sizeof(real) * n_res_var);
  for (integer b = 0; b < n_block; ++b) {
    const auto mx{mesh[b].mx}, my{mesh[b].my}, mz{mesh[b].mz};
    const integer size = mx * my * mz;
    int n_blocks = std::min(num_blocks_per_sm * num_sms, (size + TPB - 1) / TPB);
    reduction_of_dv_squared<n_res_var> <<<n_blocks, TPB, TPB * sizeof(real) * n_res_var >>>(
        field[b].h_ptr->bv_last.data(), size);
    reduction_of_dv_squared<n_res_var> <<<1, TPB, TPB * sizeof(real) * n_res_var >>>(field[b].h_ptr->bv_last.data(),
                                                                                     n_blocks);
    cudaMemcpy(res_block, field[b].h_ptr->bv_last.data(), n_res_var * sizeof(real), cudaMemcpyDeviceToHost);
    for (integer l = 0; l < n_res_var; ++l) {
      res[l] += res_block[l];
    }
  }

  if (parameter.get_bool("parallel")) {
    // Parallel reduction
    static std::array<double, 4> res_temp;
    for (int i = 0; i < 4; ++i) {
      res_temp[i] = res[i];
    }
    MPI_Allreduce(res_temp.data(), res.data(), 4, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  }
  for (auto &e: res) {
    e = std::sqrt(e / mesh.n_grid_total);
  }

  if (step == parameter.get_int("output_screen")) {
    for (integer i = 0; i < n_res_var; ++i) {
      res_scale[i] = res[i];
      if (res_scale[i] < 1e-20) {
        res_scale[i] = 1e-20;
      }
    }
    const std::filesystem::path out_dir("output/message");
    if (!exists(out_dir)) {
      create_directories(out_dir);
    }
    if (myid == 0) {
      std::ofstream res_scale_out(out_dir.string() + "/residual_scale.txt");
      res_scale_out << res_scale[0] << '\n' << res_scale[1] << '\n' << res_scale[2] << '\n' << res_scale[3] << '\n';
      res_scale_out.close();
    }
  }

  for (integer i = 0; i < 4; ++i) {
    res[i] /= res_scale[i];
  }

  // Find the maximum error of the 4 errors
  real err_max = res[0];
  for (integer i = 1; i < 4; ++i) {
    if (std::abs(res[i]) > err_max) {
      err_max = res[i];
    }
  }

  if (myid == 0) {
    if (isnan(err_max)) {
      printf("Nan occurred in step %d. Stop simulation.\n", step);
      cfd::MpiParallel::exit();
    }
  }

  return err_max;
}

template<MixtureModel mix_model, TurbMethod turb_method>
void Driver<mix_model, turb_method>::steady_screen_output(integer step, real err_max) {
  time.get_elapsed_time();
  std::ofstream history("history.dat", std::ios::app);
  history << step << '\t' << err_max << '\n';
  history.close();

  printf("\n%38s    converged to: %11.4e\n", "rho", res[0]);
  printf("  n=%8d,                       V     converged to: %11.4e   \n", step, res[1]);
  printf("  n=%8d,                       p     converged to: %11.4e   \n", step, res[2]);
  printf("%38s    converged to: %11.4e\n", "T ", res[3]);
  printf("CPU time for this step is %16.8fs\n", time.step_time);
  printf("Total elapsed CPU time is %16.8fs\n", time.elapsed_time);
//  std::cout << std::format("\n{:>38}    converged to: {:>11.4e}\n", "rho", res[0]);
//  std::cout << std::format("  n={:>8},                       V     converged to: {:>11.4e}   \n", step, res[1]);
//  std::cout << std::format("  n={:>8},                       p     converged to: {:>11.4e}   \n", step, res[2]);
//  std::cout << std::format("{:>38}    converged to: {:>11.4e}\n", "T ", res[3]);
//  std::cout << std::format("CPU time for this step is {:>16.8f}s\n", time.step_time);
//  std::cout << std::format("Total elapsed CPU time is {:>16.8f}s\n", time.elapsed_time);
}

template<MixtureModel mix_model, TurbMethod turb_method>
void Driver<mix_model, turb_method>::post_process() {
  static const std::vector<integer> processes{parameter.get_int_array("post_process")};

  for (auto process:processes){
    switch (process) {
      case 0: // Compute the 2D cf/qw
        wall_friction_heatflux_2d(mesh, field, parameter);
        break;
      default:break;
    }
  }
}

template<integer N>
__global__ void reduction_of_dv_squared(real *arr, integer size) {
  integer i = blockDim.x * blockIdx.x + threadIdx.x;
  const integer t = threadIdx.x;
  extern __shared__ real s[];
  memset(&s[t * N], 0, N * sizeof(real));
  if (i >= size) {
    return;
  }
  real inp[N];
  memset(inp, 0, N * sizeof(real));
  for (integer idx = i; idx < size; idx += blockDim.x * gridDim.x) {
    inp[0] += arr[idx];
    inp[1] += arr[idx + size];
    inp[2] += arr[idx + size * 2];
    inp[3] += arr[idx + size * 3];
  }
  for (integer l = 0; l < N; ++l) {
    s[t * N + l] = inp[l];
  }
  __syncthreads();

  for (int stride = blockDim.x / 2, lst = blockDim.x & 1; stride >= 1; lst = stride & 1, stride >>= 1) {
    stride += lst;
    __syncthreads();
    if (t < stride) {
      //when t+stride is larger than #elements, there's no meaning of comparison. So when it happens, just keep the current value for parMax[t]. This always happens when an odd number of t satisfying the condition.
      if (t + stride < size) {
        for (integer l = 0; l < N; ++l) {
          s[t * N + l] += s[(t + stride) * N + l];
        }
      }
    }
    __syncthreads();
  }

  if (t == 0) {
    arr[blockIdx.x] = s[0];
    arr[blockIdx.x + gridDim.x] = s[1];
    arr[blockIdx.x + gridDim.x * 2] = s[2];
    arr[blockIdx.x + gridDim.x * 3] = s[3];
  }
}

__global__ void compute_wall_distance(const real *wall_point_coor, DZone *zone, integer n_point_times3) {
  const integer ngg{zone->ngg}, mx{zone->mx}, my{zone->my}, mz{zone->mz};
  integer i = (integer) (blockDim.x * blockIdx.x + threadIdx.x) - ngg;
  integer j = (integer) (blockDim.y * blockIdx.y + threadIdx.y) - ngg;
  integer k = (integer) (blockDim.z * blockIdx.z + threadIdx.z) - ngg;
  if (i >= mx + ngg || j >= my + ngg || k >= mz + ngg) return;

  const real x{zone->x(i, j, k)}, y{zone->y(i, j, k)}, z{zone->z(i, j, k)};
  const integer n_wall_point = n_point_times3 / 3;
  auto &wall_dist = zone->wall_distance(i, j, k);
  wall_dist = 1e+6;
  for (integer l = 0; l < n_wall_point; ++l) {
    const integer idx = 3 * l;
    real d = (x - wall_point_coor[idx]) * (x - wall_point_coor[idx]) +
             (y - wall_point_coor[idx + 1]) * (y - wall_point_coor[idx + 1]) +
             (z - wall_point_coor[idx + 2]) * (z - wall_point_coor[idx + 2]);
    if (wall_dist > d) {
      wall_dist = d;
    }
  }
  wall_dist = std::sqrt(wall_dist);
}

// Instantiate all possible drivers
template
struct Driver<MixtureModel::Air, TurbMethod::Laminar>;
template
struct Driver<MixtureModel::Air, TurbMethod::RANS>;
template
struct Driver<MixtureModel::Mixture, TurbMethod::Laminar>;
template
struct Driver<MixtureModel::Mixture, TurbMethod::RANS>;
template
struct Driver<MixtureModel::FR, TurbMethod::Laminar>;
template
struct Driver<MixtureModel::FR, TurbMethod::RANS>;

} // cfd