#pragma once

#include "Define.h"
#include "gxl_lib/Array.hpp"
#include "gxl_lib/Matrix.hpp"
#include <vector>

namespace cfd {
struct Boundary {
  Boundary() = default;

  Boundary(integer x1, integer x2, integer y1, integer y2, integer z1,
           integer z2, integer type);

  /**
   * \brief process the boundary information to identify its face and direction.
   * \param ngg number of ghost grid in each direction
   * \param dim dimension of the problem
   */
  void register_boundary(integer ngg, integer dim);

  /** the coordinate range of 3 directions*/
  integer range_start[3] = {0, 0, 0};
  integer range_end[3] = {0, 0, 0};
  /** type identifier for the boundary*/
  integer type_label = 0;
  /**> the normal direction of the face, i-0, j-1, k-2*/
  integer face = 0;
  /**> is the face normal positive or negative. Default: +1, which means the
   * normal points to the direction that the coordinate increases*/
  integer direction = 1;
};

struct InnerFace {
  InnerFace(integer x1, integer x2, integer y1, integer y2, integer z1,
            integer z2, integer tx1, integer tx2, integer ty1, integer ty2,
            integer tz1, integer tz2, integer block_id);

  /**
   * \brief establish the corresponding relation of the faces
   * \param ngg number of ghost grids in each direction
   * \param dim dimension
   */
  void register_boundary(integer ngg, integer dim);

  integer range_start[3]{0, 0, 0};
  integer range_end[3]{0, 0, 0};  // the coordinate range of 3 directions
  integer face = 0;  // the normal direction of the face, i-0, j-1, k-2
  // is the face normal positive or negative. Default: +1, which means the
  // normal points to the direction that the coordinate increases
  integer direction = 1;
  integer target_start[3]{0, 0, 0},
      target_end[3]{0, 0, 0};  // The coordinate range of target block
  integer target_block = 0;                // The target block number.
  integer target_face = 0;  // the normal direction of the target face, i-0, j-1, k-2
  integer target_direction = 1;  // is the target face normal positive or negative. Default: +1
  integer src_tar[3]{0, 0, 0};  // the corresponding relation between this source face and the target face
  integer loop_dir[3]{1, 1, 1};  // The direction that when looping over the face, i,j,k increment +1/-1
  integer target_loop_dir[3]{1, 1, 1};  // The direction that when looping over the face, i,j,k increment +1/-1
  integer n_point[3]{0, 0, 0};
};

struct ParallelFace {
  ParallelFace(integer x1, integer x2, integer y1, integer y2, integer z1,
               integer z2, integer proc_id, integer flag_s, integer flag_r);

  ParallelFace() = default;

  /**
   * \brief establish the value passing order of the face.
   * \param dim dimension
   * \param ngg number of ghost grids in each direction
   * \details The faces are stored in a fixed order @loop_order. The first face
   * is the matched face, the second one is the face with positive index and the
   * last face is the one with negative face.
   */
  void register_boundary(integer dim, integer ngg);

  integer range_start[3]{0, 0, 0};  // The index of starting point in 3 directions of the current face.
  integer range_end[3]{0, 0, 0};  // The index of ending point in 3 directions of the current face.
  integer face = 0;  // the normal direction of the face, i-0, j-1, k-2.
  /**
   * \brief The face normal positive or negative.
   * Default: +1, which means the normal points to the direction that the
   * coordinate increases.
   */
  integer direction = 1;
  integer target_process = 0;  // The target block number.
  integer flag_send = 0, flag_receive = 0;
  /**
   * \brief When sending message, the order of data put into the buffer.
   * \details The label with same number is first iterated, then the positive
   * one, and the negative direction last. When getting data out of the
   * received, also in the same order.
   */
  integer loop_order[3]{0, 0, 0};
  /**
   * \brief The direction for iteration of each coordinates, +1/-1
   * \details The order is not directly in i/j/k directions, but in the order of
   * @loop_order.
   */
  integer loop_dir[3]{1, 1, 1};
};

/**
 * \brief The class of a grid block.
 * \details A geometric class, all contained messages are about geometric
 * information.
 */
class Block {
public:
  explicit Block(integer _mx, integer _my, integer _mz,
                 integer _ngg, integer _id);

  /**
   * \brief compute the jacobian and metric matrix of the current block
   * \param myid process number
   */
  void compute_jac_metric(integer myid);

  void trim_abundant_ghost_mesh();

private:
  /**
    * \brief create the header of the error log about negative jacobians.
    * \param myid current process id
    */
  void log_negative_jacobian(integer myid, integer i, integer j, integer k) const;

public:
  integer mx = 1, my = 1, mz = 1;
  integer n_grid = 1;
  integer block_id = 0;
  integer ngg = 2;
  gxl::Array3D<real> x, y, z;
  std::vector<Boundary> boundary;
  std::vector<InnerFace> inner_face;
  std::vector<ParallelFace> parallel_face;
  gxl::Array3D<real> jacobian;
  /**
   * \brief array of metrics of the grid points.
   * \details The metric matrix consists of
   *         \f[
   *         \left[\begin{array}{ccc}
   *             \xi_x  &\xi_y  &\xi_z \\
   *             \eta_x &\eta_y &\eta_z \\
   *             \zeta_x&\zeta_y&\zeta_z
   *             \end{array}\right]
   *             \f]
   */
  gxl::Array3D<gxl::Matrix<real, 3, 3, 1>> metric;
#ifdef GPU
//  ggxl::Array3D<real> d_x, d_y, d_z;
//  Boundary *d_bound;
//  InnerFace *d_innerface;
//  ParallelFace *d_parface;
//  ggxl::Array3D<real> d_jac;
//  ggxl::Array3D<gxl::Matrix<real, 3, 3, 1>> d_metric;
#endif
};

class Parameter;

class Mesh {
public:
  integer dimension = 3;
  integer n_block = 1;
  integer n_grid = 1;
  integer n_grid_total = 1;
private:
  std::vector<Block> block;

public:
  explicit Mesh(Parameter &parameter);

  Block &operator[](size_t i);

  const Block &operator[](size_t i) const;

private:
  void read_grid(integer myid, integer ngg);

  /**
   * \brief read the physical boundary of current process
   * \param myid the process id of current process
   * \param ngg number of ghost layers
   */
  void read_boundary(integer myid, integer ngg);

  /**
   * \brief read the inner face communication message of current process
   * \param myid the process id of current process
   * \param ngg number of ghost layers
   */
  void read_inner_interface(integer myid, integer ngg);

  /**
   * \brief read the parallel boundary coordinates. Do not read the target face or match them, left for solver initialization
   * \param myid process number, used for identify which file to read
   * \param ngg number of ghost grids in each direction
   */
  void read_parallel_interface(integer myid, integer ngg);

  /**
   * \brief scale all coordinates (x/y/z) to unit of meters.
   * \param scale the scale of the coordinates
   * \details for example, if the grid is drawn in unit mm, when we compute it in meters, it should be multiplied by 0.001
   *  first, where the 0.001 is the @scale here.
   */
  void scale(real scale);

  /**
   * \brief initialize the ghost grids of the simulation
   * \param myid process number, used for identify which file to read
   * \param parallel if the computation is conducted in parallel
   * \param ngg number of ghost layers
   */
  void init_ghost_grid(integer myid, bool parallel, integer ngg);

  /**
   * \brief called by @init_ghost_grid, used for initializing ghost grids of the inner faces
   */
  void init_inner_ghost_grid();

  /**
   * \brief called by @init_ghost_grid, initialize the ghost grids of parallel communication faces
   * \param myid process number, used for identify which file to read
   * \param ngg number of ghost layers
   */
  void init_parallel_ghost_grid(integer myid, integer ngg);

//  void copy_mesh_to_device();
};
}
