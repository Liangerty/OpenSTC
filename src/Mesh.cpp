#include "Mesh.h"
#include "gxl_lib/Math.hpp"
#include "fmt/format.h"
#include <mpi.h>
#include <filesystem>
#include <fstream>
#include "Parameter.h"
#include <sstream>

cfd::Boundary::Boundary(integer x1, integer x2, integer y1, integer y2,
                        integer z1, integer z2, integer type)
    : range_start{x1, y1, z1}, range_end{x2, y2, z2}, type_label{type} {}

void cfd::Boundary::register_boundary(const integer ngg, const integer dim) {
  for (integer i = 0; i < dim; ++i) {
    if (range_start[i] == range_end[i]) {
      face = i;
      if (range_start[i] == 0) direction = -1;
      break;
    }
  }
  for (integer i = 0; i < dim; ++i) {
    if (i == face) continue;
    if (range_end[i] < range_start[i]) std::swap(range_end[i], range_start[i]);
    range_start[i] -= ngg;
    range_end[i] += ngg;
  }
  if (dim == 2) {
    range_start[2] = -ngg;
    range_end[2] = ngg;
  }
}

cfd::InnerFace::InnerFace(integer x1, integer x2, integer y1, integer y2,
                          integer z1, integer z2, integer tx1, integer tx2,
                          integer ty1, integer ty2, integer tz1, integer tz2,
                          integer block_id)
    : range_start{x1, y1, z1},
      range_end{x2, y2, z2},
      target_start{tx1, ty1, tz1},
      target_end{tx2, ty2, tz2},
      target_block{block_id - 1} {}

void cfd::InnerFace::register_boundary(const integer ngg, const integer dim) {
  for (integer i = 0; i < dim; ++i) {
    // if start==end, then it is the normal direction
    if (range_start[i] == range_end[i]) {
      face = i;
      // if the same value is 1, it is the small label face, and the normal
      // points to negative direction
      if (range_start[i] == 1) direction = -1;
      range_start[i] -= 1;
      range_end[i] -= 1;
      break;
    }
  }
  for (integer i = 0; i < dim; ++i) {
    if (target_start[i] == target_end[i]) {
      target_face = i;
      if (target_start[i] == 1) target_direction = -1;
      target_start[i] -= 1;
      target_end[i] -= 1;
      break;
    }
  }
  src_tar[target_face] = -(direction * target_direction) * (face + 1);
  if (dim == 2) {
    const integer next_face{1 - face}, target_next_face{1 - target_face};
    range_start[next_face] = std::abs(range_start[next_face]) - 1;
    range_end[next_face] = std::abs(range_end[next_face]) - 1;
    target_start[next_face] = std::abs(target_start[next_face]) - 1;
    target_end[next_face] = std::abs(target_end[next_face]) - 1;
    src_tar[target_next_face] = gxl::sgn((target_end[target_next_face] - target_start[target_next_face]) *
                                         (range_end[next_face] - range_start[next_face])) * (next_face + 1);
    src_tar[2] = 3;
    range_start[2] = 0;
    range_end[2] = 0;
    target_start[2] = 0;
    target_end[2] = 0;
  } else {
    integer next_face{0}, target_next_face{0};
    for (integer i = 0; i < dim; ++i) {
      if (range_start[i] < 0) {
        next_face = i;
        range_start[i] = -range_start[i] - 1;
        range_end[i] = -range_end[i] - 1;
        break;
      }
    }
    for (integer i = 0; i < dim; ++i) {
      if (target_start[i] < 0) {
        target_next_face = i;
        target_start[i] = -target_start[i] - 1;
        target_end[i] = -target_end[i] - 1;
        break;
      }
    }
    src_tar[target_next_face] = gxl::sgn((target_end[target_next_face] - target_start[target_next_face]) *
                                         (range_end[next_face] - range_start[next_face])) * (next_face + 1);
    next_face = 3 - face - next_face;
    target_next_face = 3 - target_face - target_next_face;
    range_start[next_face] -= 1;
    range_end[next_face] -= 1;
    target_start[target_next_face] -= 1;
    target_end[target_next_face] -= 1;
    src_tar[target_next_face] = gxl::sgn((target_end[target_next_face] - target_start[target_next_face]) *
                                         (range_end[next_face] - range_start[next_face])) * (next_face + 1);
  }

  // Here, the range of loop for corresponding face is changed to the ghost grid
  // range. The other two directions are also changed, which means the current
  // method also communicate the data in corners. ACANS and Wang's code
  // communicate the corners in default. Later if we do not want to communicate
  // corner data in default, the ranges for the other 2 directions should be
  // decreased, which can be achieved by just commenting out the last loop of
  // this function.
  range_start[face] = direction < 0 ? -ngg : range_end[face];
  range_end[face] = direction < 0 ? 0 : range_end[face] + ngg;
  const integer sgn_tar = gxl::sgn(src_tar[target_face]);
  if (target_direction > 0) {
    if (sgn_tar > 0) {
      target_start[target_face] = target_end[target_face] - ngg;
    } else {
      target_start[target_face] = target_end[target_face];
    }
  } else {
    target_start[target_face] = (sgn_tar > 0) ? 0 : ngg;
  }
  for (integer i = 0; i < 3; ++i) {
    loop_dir[i] = range_start[i] <= range_end[i] ? 1 : -1;
    target_loop_dir[i] = gxl::sgn(src_tar[i]);
    src_tar[i] = std::abs(src_tar[i]) - 1;
  }
  // Include the corners into the transfer
  // Commented on 2023/6/20. The data on corneres are not communicated
  /*
   * For corners such as
   *
   *                    |
   *                    |
   *                    |
   *                    |
   *                    |
   * -------------------*
   *                    :
   *                    :
   *                    :
   *                    :
   *
   * which consists of a vertical wall(|),
   *                   a horizontal wall(-)
   *               and a vertical inner boundary(:)
   *             below the corner point(*).
   * We name the left side block 0, which contains the horizontal line and the fluid below,
   * the right side block 1, which contains the vertical line and the fuild on its right.
   *
   * If the communication contains the ghost grids, then for block 1,
   * the communicated region would contain not only the vertical inner boundary,
   * but also 2 points above the corner point.
   * Those 2 points above the corner point is assigned to be the average between the 2 blocks.
   * For this scene, the value of those 2 points are 0 in block 1 because they are on the wall.
   * Those values on block 0 is the negative of the inner 2 points below the horizontal wall
   * as they are ghost grids in block 0. Thus, when we average the 2 values, we get non-zero value on these 2 points,
   * which disobeys the wall bondary condition.
   *
   * */
//  for (integer i = 0; i < 3; ++i) {
//    if (i != face) {
//      range_start[i] -= loop_dir[i] * ngg;
//      range_end[i] += loop_dir[i] * ngg;
//      target_start[i] -= target_loop_dir[i] * loop_dir[i] * ngg;
//      target_end[i] += target_loop_dir[i] * loop_dir[i] * ngg;
//    }
//  }
  for (integer i = 0; i < 3; ++i) {
    n_point[i] = abs(range_start[i] - range_end[i]) + 1;
  }
}

cfd::ParallelFace::ParallelFace(integer x1, integer x2, integer y1, integer y2,
                                integer z1, integer z2, integer proc_id,
                                integer flag_s, integer flag_r)
    : range_start{x1, y1, z1},
      range_end{x2, y2, z2},
      target_process{proc_id},
      flag_send{flag_s},
      flag_receive{flag_r} {}

void cfd::ParallelFace::register_boundary(const integer dim, integer ngg) {
  for (integer i = 0; i < dim; ++i) {
    // if start==end, then it is the normal direction
    if (range_start[i] == range_end[i]) {
      face = i;
      // if the same value is 1, it is the small label face, and the normal
      // points to negative direction
      if (range_start[i] == 1) direction = -1;
      range_start[i] -= 1;
      range_end[i] -= 1;
      loop_order[0] = i;
      loop_dir[0] = 0;
      break;
    }
  }
  if (dim == 2) {
    const integer next_face{1 - face};
    range_start[next_face] = std::abs(range_start[next_face]) - 1;
    range_end[next_face] = std::abs(range_end[next_face]) - 1;
    loop_order[1] = next_face;
    loop_dir[1] = range_start[next_face] < range_end[next_face] ? 1 : -1;
    range_start[2] = 0;
    range_end[2] = 0;
    loop_order[2] = 2;
    loop_dir[2] = 0;
  } else {
    integer next_face{0};
    for (integer i = 0; i < dim; ++i) {
      if (range_start[i] < 0) {
        next_face = i;
        range_start[i] = -range_start[i] - 1;
        range_end[i] = -range_end[i] - 1;
        loop_order[2] = i;
        loop_dir[2] = range_start[i] < range_end[i] ? 1 : -1;
        break;
      }
    }
    next_face = 3 - face - next_face;
    range_start[next_face] -= 1;
    range_end[next_face] -= 1;
    loop_order[1] = next_face;
    loop_dir[1] = range_start[next_face] < range_end[next_face] ? 1 : -1;
  }
  // Include the corner into the transfer
  // Commented
//  for (integer i = 0; i < 3; ++i) {
//    if (i != face) {
//      range_start[i] -= loop_dir[i] * ngg;
//      range_end[i] += loop_dir[i] * ngg;
//    }
//  }
  for (integer i = 0; i < 3; ++i) {
    n_point[i] = abs(range_start[i] - range_end[i]) + 1;
  }
}

cfd::Block::Block(const integer _mx, const integer _my, const integer _mz,
                  const integer _ngg, const integer _id)
    : mx{_mx},
      my{_my},
      mz{_mz},
      n_grid{mx * my * mz},
      block_id{_id},
      ngg{_ngg},
      x{mx, my, mz, _ngg + 1},
      y{mx, my, mz, _ngg + 1},
      z{mx, my, mz, _ngg + 1},
      jacobian{mx, my, mz, _ngg},
      metric{mx, my, mz, _ngg} {}

void cfd::Block::compute_jac_metric(integer myid) {
  // First the inner part is computed, excluding the boundaries and ghost grids
  for (integer i = 1; i < mx - 1; ++i) {
    for (integer j = 1; j < my - 1; ++j) {
      integer k_min{1}, k_max{mz - 1};
      if (mz == 1) {
        // which means dimension = 2
        k_min = 0;
        k_max = 1;
      }
      for (integer k = k_min; k < k_max; ++k) {
        const real dxd1 = (x(i + 1, j, k) - x(i - 1, j, k)) * 0.5,
            dyd1 = (y(i + 1, j, k) - y(i - 1, j, k)) * 0.5,
            dzd1 = (z(i + 1, j, k) - z(i - 1, j, k)) * 0.5;
        const real dxd2 = (x(i, j + 1, k) - x(i, j - 1, k)) * 0.5,
            dyd2 = (y(i, j + 1, k) - y(i, j - 1, k)) * 0.5,
            dzd2 = (z(i, j + 1, k) - z(i, j - 1, k)) * 0.5;
        const real dxd3 = (x(i, j, k + 1) - x(i, j, k - 1)) * 0.5,
            dyd3 = (y(i, j, k + 1) - y(i, j, k - 1)) * 0.5,
            dzd3 = (z(i, j, k + 1) - z(i, j, k - 1)) * 0.5;
        const real jac = dxd1 * dyd2 * dzd3 + dxd2 * dyd3 * dzd1 +
                         dxd3 * dyd1 * dzd2 - dxd1 * dyd3 * dzd2 -
                         dxd2 * dyd1 * dzd3 - dxd3 * dyd2 * dzd1;
        if (jac <= 0) {
          fmt::print(
              "Negative Jacobian from process {}\nBlock {}, index is ({}, {}, "
              "{}).\nStop simulation.\n",
              myid, block_id, i, j, k);
          MPI_Abort(MPI_COMM_WORLD, 1);
        }
        jacobian(i, j, k) = jac;
        auto &m = metric(i, j, k);
        m(1, 1) = (dyd2 * dzd3 - dyd3 * dzd2) / jac;
        m(1, 2) = (dxd3 * dzd2 - dxd2 * dzd3) / jac;
        m(1, 3) = (dxd2 * dyd3 - dxd3 * dyd2) / jac;
        m(2, 1) = (dyd3 * dzd1 - dyd1 * dzd3) / jac;
        m(2, 2) = (dxd1 * dzd3 - dxd3 * dzd1) / jac;
        m(2, 3) = (dxd3 * dyd1 - dxd1 * dyd3) / jac;
        m(3, 1) = (dyd1 * dzd2 - dyd2 * dzd1) / jac;
        m(3, 2) = (dxd2 * dzd1 - dxd1 * dzd2) / jac;
        m(3, 3) = (dxd1 * dyd2 - dxd2 * dyd1) / jac;
      }
    }
  }
  // Start to compute boundaries, edges and corners. If negative value appears
  // here, just assign the value of the inner layer to it. First block:
  // k=-1,-ng No edges and corners
  for (integer i = 1; i < mx - 1; ++i) {
    for (integer j = 1; j < my - 1; ++j) {
      for (integer k = 0; k >= -ngg; --k) {
        const double dxd1 = (x(i + 1, j, k) - x(i - 1, j, k)) * 0.5,
            dyd1 = (y(i + 1, j, k) - y(i - 1, j, k)) * 0.5,
            dzd1 = (z(i + 1, j, k) - z(i - 1, j, k)) * 0.5;
        const double dxd2 = (x(i, j + 1, k) - x(i, j - 1, k)) * 0.5,
            dyd2 = (y(i, j + 1, k) - y(i, j - 1, k)) * 0.5,
            dzd2 = (z(i, j + 1, k) - z(i, j - 1, k)) * 0.5;
        const double dxd3 = (x(i, j, k + 1) - x(i, j, k - 1)) * 0.5,
            dyd3 = (y(i, j, k + 1) - y(i, j, k - 1)) * 0.5,
            dzd3 = (z(i, j, k + 1) - z(i, j, k - 1)) * 0.5;
        const double jac = dxd1 * dyd2 * dzd3 + dxd2 * dyd3 * dzd1 +
                           dxd3 * dyd1 * dzd2 - dxd1 * dyd3 * dzd2 -
                           dxd2 * dyd1 * dzd3 - dxd3 * dyd2 * dzd1;
        if (jac <= 0) {
          log_negative_jacobian(myid, i, j, k);
          jacobian(i, j, k) = jacobian(i, j, k + 1);
          metric(i, j, k) = metric(i, j, k + 1);
          continue;
        }
        jacobian(i, j, k) = jac;
        auto &m = metric(i, j, k);
        m(1, 1) = (dyd2 * dzd3 - dyd3 * dzd2) / jac;
        m(1, 2) = (dxd3 * dzd2 - dxd2 * dzd3) / jac;
        m(1, 3) = (dxd2 * dyd3 - dxd3 * dyd2) / jac;
        m(2, 1) = (dyd3 * dzd1 - dyd1 * dzd3) / jac;
        m(2, 2) = (dxd1 * dzd3 - dxd3 * dzd1) / jac;
        m(2, 3) = (dxd3 * dyd1 - dxd1 * dyd3) / jac;
        m(3, 1) = (dyd1 * dzd2 - dyd2 * dzd1) / jac;
        m(3, 2) = (dxd2 * dzd1 - dxd1 * dzd2) / jac;
        m(3, 3) = (dxd1 * dyd2 - dxd2 * dyd1) / jac;
      }
    }
  }
  // Second block: k=mz,mz+ngg-1, No edges and corners
  for (integer i = 1; i < mx - 1; ++i) {
    for (integer j = 1; j < my - 1; ++j) {
      for (integer k = mz - 1; k < mz + ngg; ++k) {
        const double dxd1 = (x(i + 1, j, k) - x(i - 1, j, k)) / 2,
            dyd1 = (y(i + 1, j, k) - y(i - 1, j, k)) / 2,
            dzd1 = (z(i + 1, j, k) - z(i - 1, j, k)) / 2;
        const double dxd2 = (x(i, j + 1, k) - x(i, j - 1, k)) / 2,
            dyd2 = (y(i, j + 1, k) - y(i, j - 1, k)) / 2,
            dzd2 = (z(i, j + 1, k) - z(i, j - 1, k)) / 2;
        const double dxd3 = (x(i, j, k + 1) - x(i, j, k - 1)) / 2,
            dyd3 = (y(i, j, k + 1) - y(i, j, k - 1)) / 2,
            dzd3 = (z(i, j, k + 1) - z(i, j, k - 1)) / 2;
        const double jac = dxd1 * dyd2 * dzd3 + dxd2 * dyd3 * dzd1 +
                           dxd3 * dyd1 * dzd2 - dxd1 * dyd3 * dzd2 -
                           dxd2 * dyd1 * dzd3 - dxd3 * dyd2 * dzd1;
        if (jac <= 0) {
          log_negative_jacobian(myid, i, j, k);
          jacobian(i, j, k) = jacobian(i, j, k - 1);
          metric(i, j, k) = metric(i, j, k - 1);
          continue;
        }
        jacobian(i, j, k) = jac;
        auto &m = metric(i, j, k);
        m(1, 1) = (dyd2 * dzd3 - dyd3 * dzd2) / jac;
        m(1, 2) = (dxd3 * dzd2 - dxd2 * dzd3) / jac;
        m(1, 3) = (dxd2 * dyd3 - dxd3 * dyd2) / jac;
        m(2, 1) = (dyd3 * dzd1 - dyd1 * dzd3) / jac;
        m(2, 2) = (dxd1 * dzd3 - dxd3 * dzd1) / jac;
        m(2, 3) = (dxd3 * dyd1 - dxd1 * dyd3) / jac;
        m(3, 1) = (dyd1 * dzd2 - dyd2 * dzd1) / jac;
        m(3, 2) = (dxd2 * dzd1 - dxd1 * dzd2) / jac;
        m(3, 3) = (dxd1 * dyd2 - dxd2 * dyd1) / jac;
      }
    }
  }
  // Third block: j=-1,-ngg, with corners
  for (integer i = 1; i < mx - 1; ++i) {
    for (integer j = 0; j >= -ngg; --j) {
      for (integer k = -ngg; k < mz + ngg; ++k) {
        double dxd1 = (x(i + 1, j, k) - x(i - 1, j, k)) / 2,
            dyd1 = (y(i + 1, j, k) - y(i - 1, j, k)) / 2,
            dzd1 = (z(i + 1, j, k) - z(i - 1, j, k)) / 2;
        double dxd2 = (x(i, j + 1, k) - x(i, j - 1, k)) / 2,
            dyd2 = (y(i, j + 1, k) - y(i, j - 1, k)) / 2,
            dzd2 = (z(i, j + 1, k) - z(i, j - 1, k)) / 2;
        double dxd3 = (x(i, j, k + 1) - x(i, j, k - 1)) / 2,
            dyd3 = (y(i, j, k + 1) - y(i, j, k - 1)) / 2,
            dzd3 = (z(i, j, k + 1) - z(i, j, k - 1)) / 2;
        double jac = dxd1 * dyd2 * dzd3 + dxd2 * dyd3 * dzd1 +
                     dxd3 * dyd1 * dzd2 - dxd1 * dyd3 * dzd2 -
                     dxd2 * dyd1 * dzd3 - dxd3 * dyd2 * dzd1;
        if (jac <= 0) {
          log_negative_jacobian(myid, i, j, k);
          jacobian(i, j, k) = jacobian(i, j + 1, k);
          metric(i, j, k) = metric(i, j + 1, k);
          continue;
        }
        jacobian(i, j, k) = jac;
        auto &m = metric(i, j, k);
        m(1, 1) = (dyd2 * dzd3 - dyd3 * dzd2) / jac;
        m(1, 2) = (dxd3 * dzd2 - dxd2 * dzd3) / jac;
        m(1, 3) = (dxd2 * dyd3 - dxd3 * dyd2) / jac;
        m(2, 1) = (dyd3 * dzd1 - dyd1 * dzd3) / jac;
        m(2, 2) = (dxd1 * dzd3 - dxd3 * dzd1) / jac;
        m(2, 3) = (dxd3 * dyd1 - dxd1 * dyd3) / jac;
        m(3, 1) = (dyd1 * dzd2 - dyd2 * dzd1) / jac;
        m(3, 2) = (dxd2 * dzd1 - dxd1 * dzd2) / jac;
        m(3, 3) = (dxd1 * dyd2 - dxd2 * dyd1) / jac;
      }
    }
  }
  // Fourth block: j=my,my+ngg-1, with edges and corners
  for (integer i = 1; i < mx - 1; ++i) {
    for (integer j = my - 1; j < my + ngg; ++j) {
      for (integer k = -ngg; k < mz + ngg; ++k) {
        double dxd1 = (x(i + 1, j, k) - x(i - 1, j, k)) / 2,
            dyd1 = (y(i + 1, j, k) - y(i - 1, j, k)) / 2,
            dzd1 = (z(i + 1, j, k) - z(i - 1, j, k)) / 2;
        double dxd2 = (x(i, j + 1, k) - x(i, j - 1, k)) / 2,
            dyd2 = (y(i, j + 1, k) - y(i, j - 1, k)) / 2,
            dzd2 = (z(i, j + 1, k) - z(i, j - 1, k)) / 2;
        double dxd3 = (x(i, j, k + 1) - x(i, j, k - 1)) / 2,
            dyd3 = (y(i, j, k + 1) - y(i, j, k - 1)) / 2,
            dzd3 = (z(i, j, k + 1) - z(i, j, k - 1)) / 2;
        double jac = dxd1 * dyd2 * dzd3 + dxd2 * dyd3 * dzd1 +
                     dxd3 * dyd1 * dzd2 - dxd1 * dyd3 * dzd2 -
                     dxd2 * dyd1 * dzd3 - dxd3 * dyd2 * dzd1;
        if (jac <= 0) {
          log_negative_jacobian(myid, i, j, k);
          jacobian(i, j, k) = jacobian(i, j - 1, k);
          metric(i, j, k) = metric(i, j - 1, k);
          continue;
        }
        jacobian(i, j, k) = jac;
        auto &m = metric(i, j, k);
        m(1, 1) = (dyd2 * dzd3 - dyd3 * dzd2) / jac;
        m(1, 2) = (dxd3 * dzd2 - dxd2 * dzd3) / jac;
        m(1, 3) = (dxd2 * dyd3 - dxd3 * dyd2) / jac;
        m(2, 1) = (dyd3 * dzd1 - dyd1 * dzd3) / jac;
        m(2, 2) = (dxd1 * dzd3 - dxd3 * dzd1) / jac;
        m(2, 3) = (dxd3 * dyd1 - dxd1 * dyd3) / jac;
        m(3, 1) = (dyd1 * dzd2 - dyd2 * dzd1) / jac;
        m(3, 2) = (dxd2 * dzd1 - dxd1 * dzd2) / jac;
        m(3, 3) = (dxd1 * dyd2 - dxd2 * dyd1) / jac;
      }
    }
  }
  // Fifth block: i=-1,-ngg, with corners
  for (integer i = 0; i >= -ngg; --i) {
    for (integer j = -ngg; j < my + ngg; ++j) {
      for (integer k = -ngg; k < mz + ngg; ++k) {
        double dxd1 = (x(i + 1, j, k) - x(i - 1, j, k)) / 2,
            dyd1 = (y(i + 1, j, k) - y(i - 1, j, k)) / 2,
            dzd1 = (z(i + 1, j, k) - z(i - 1, j, k)) / 2;
        double dxd2 = (x(i, j + 1, k) - x(i, j - 1, k)) / 2,
            dyd2 = (y(i, j + 1, k) - y(i, j - 1, k)) / 2,
            dzd2 = (z(i, j + 1, k) - z(i, j - 1, k)) / 2;
        double dxd3 = (x(i, j, k + 1) - x(i, j, k - 1)) / 2,
            dyd3 = (y(i, j, k + 1) - y(i, j, k - 1)) / 2,
            dzd3 = (z(i, j, k + 1) - z(i, j, k - 1)) / 2;
        double jac = dxd1 * dyd2 * dzd3 + dxd2 * dyd3 * dzd1 +
                     dxd3 * dyd1 * dzd2 - dxd1 * dyd3 * dzd2 -
                     dxd2 * dyd1 * dzd3 - dxd3 * dyd2 * dzd1;
        if (jac <= 0) {
          log_negative_jacobian(myid, i, j, k);
          jacobian(i, j, k) = jacobian(i + 1, j, k);
          metric(i, j, k) = metric(i + 1, j, k);
          continue;
        }
        jacobian(i, j, k) = jac;
        auto &m = metric(i, j, k);
        m(1, 1) = (dyd2 * dzd3 - dyd3 * dzd2) / jac;
        m(1, 2) = (dxd3 * dzd2 - dxd2 * dzd3) / jac;
        m(1, 3) = (dxd2 * dyd3 - dxd3 * dyd2) / jac;
        m(2, 1) = (dyd3 * dzd1 - dyd1 * dzd3) / jac;
        m(2, 2) = (dxd1 * dzd3 - dxd3 * dzd1) / jac;
        m(2, 3) = (dxd3 * dyd1 - dxd1 * dyd3) / jac;
        m(3, 1) = (dyd1 * dzd2 - dyd2 * dzd1) / jac;
        m(3, 2) = (dxd2 * dzd1 - dxd1 * dzd2) / jac;
        m(3, 3) = (dxd1 * dyd2 - dxd2 * dyd1) / jac;
      }
    }
  }
  // Sixth block: i=mx,mx+ngg-1, with corners and edges
  for (integer i = mx - 1; i < mx + ngg; ++i) {
    for (integer j = -ngg; j < my + ngg; ++j) {
      for (integer k = -ngg; k < mz + ngg; ++k) {
        double dxd1 = (x(i + 1, j, k) - x(i - 1, j, k)) / 2,
            dyd1 = (y(i + 1, j, k) - y(i - 1, j, k)) / 2,
            dzd1 = (z(i + 1, j, k) - z(i - 1, j, k)) / 2;
        double dxd2 = (x(i, j + 1, k) - x(i, j - 1, k)) / 2,
            dyd2 = (y(i, j + 1, k) - y(i, j - 1, k)) / 2,
            dzd2 = (z(i, j + 1, k) - z(i, j - 1, k)) / 2;
        double dxd3 = (x(i, j, k + 1) - x(i, j, k - 1)) / 2,
            dyd3 = (y(i, j, k + 1) - y(i, j, k - 1)) / 2,
            dzd3 = (z(i, j, k + 1) - z(i, j, k - 1)) / 2;
        double jac = dxd1 * dyd2 * dzd3 + dxd2 * dyd3 * dzd1 +
                     dxd3 * dyd1 * dzd2 - dxd1 * dyd3 * dzd2 -
                     dxd2 * dyd1 * dzd3 - dxd3 * dyd2 * dzd1;
        if (jac <= 0) {
          log_negative_jacobian(myid, i, j, k);
          jacobian(i, j, k) = jacobian(i - 1, j, k);
          metric(i, j, k) = metric(i - 1, j, k);
          continue;
        }
        jacobian(i, j, k) = jac;
        auto &m = metric(i, j, k);
        m(1, 1) = (dyd2 * dzd3 - dyd3 * dzd2) / jac;
        m(1, 2) = (dxd3 * dzd2 - dxd2 * dzd3) / jac;
        m(1, 3) = (dxd2 * dyd3 - dxd3 * dyd2) / jac;
        m(2, 1) = (dyd3 * dzd1 - dyd1 * dzd3) / jac;
        m(2, 2) = (dxd1 * dzd3 - dxd3 * dzd1) / jac;
        m(2, 3) = (dxd3 * dyd1 - dxd1 * dyd3) / jac;
        m(3, 1) = (dyd1 * dzd2 - dyd2 * dzd1) / jac;
        m(3, 2) = (dxd2 * dzd1 - dxd1 * dzd2) / jac;
        m(3, 3) = (dxd1 * dyd2 - dxd2 * dyd1) / jac;
      }
    }
  }
}

void cfd::Block::log_negative_jacobian(const integer myid, const integer i,
                                       const integer j, const integer k) const {
  const std::filesystem::path out_dir("error-log/negative-jacobian");
  if (!exists(out_dir)) create_directories(out_dir);
  const auto path1 = out_dir.string();
  const auto name = fmt::format("{}/Process_{}-Block_{}.log", path1, myid, block_id);
  std::ofstream err_log(name);
  err_log << fmt::format("Block {}, I = {}\tJ = {}\tK = {}\n", block_id, i, j, k);
  err_log.close();
}

void cfd::Block::trim_abundant_ghost_mesh() {
  gxl::Array3D<real> xx(x), yy(y), zz(z);
  x.resize(mx, my, mz, ngg);
  y.resize(mx, my, mz, ngg);
  z.resize(mx, my, mz, ngg);
  for (int i = -ngg; i < mx + ngg; i++) {
    for (int j = -ngg; j < my + ngg; ++j) {
      for (int k = -ngg; k < mz + ngg; k++) {
        x(i, j, k) = xx(i, j, k);
        y(i, j, k) = yy(i, j, k);
        z(i, j, k) = zz(i, j, k);
      }
    }
  }
}

cfd::Mesh::Mesh(Parameter &parameter) : dimension{parameter.get_int("dimension")}, ngg{parameter.get_int("ngg")},
                                        n_proc{parameter.get_int("n_proc")}, nblk{new integer[n_proc]} {
  const integer myid = parameter.get_int("myid");
  const bool parallel = parameter.get_bool("parallel");
  //First read the grid points into memory
  read_grid(myid/*, ngg*/);
  parameter.update_parameter("n_block", n_block);
  if (myid == 0) {
    fmt::print("Problem dimension: {}\nTotal grid number: {}\n", dimension, n_grid_total);
  }

  read_boundary(myid/*, ngg*/);

  read_inner_interface(myid/*, ngg*/);

  if (parallel) {
    read_parallel_interface(myid/*, ngg*/);
  }

  /*Scale the grid to physical units*/
  if (const auto grid_scale = parameter.get_real("gridScale"); std::abs(grid_scale - 1.0) > 1e-15) {
    scale(grid_scale);
    if (myid == 0) {
      fmt::print("Grid scale is {}m.\n", grid_scale);
    }
  }
  if (myid == 0) {
    fmt::print("Finish reading mesh information.\n");
  }

  /*Initialize the coordinates of ghost grids*/
  init_ghost_grid(myid, parallel/*, ngg*/);

  /*Compute the metrics and jacobian values of all grid points*/
  for (auto &b: block) b.compute_jac_metric(myid);
  if (myid == 0) fmt::print("Finish computing metrics and jacobian values.\n");

  for (auto &b: block) b.trim_abundant_ghost_mesh();
}

cfd::Block &cfd::Mesh::operator[](const size_t i) {
  return block[i];
}

const cfd::Block &cfd::Mesh::operator[](const size_t i) const {
  return block[i];
}

void cfd::Mesh::read_grid(const integer myid/*, const integer ngg*/) {
  std::ifstream grd(fmt::format("./input_files/grid/grid{:>4}.grd", myid), std::ios::in);
  std::string input;
  std::getline(grd, input);
  n_block = std::stoi(input); //	Read number of grid blocks
  for (integer i = 0; i < n_block; ++i) {
    //	Read grid number in three directions of each block
    std::getline(grd, input);
    std::istringstream line(input);
    integer mx = 0, my = 0, mz = 0;
    line >> mx >> my >> mz;
    block.emplace_back(mx, my, mz, ngg, i);
  }
  if (block[0].mz == 1) {
    dimension = 2;
  }
  block.shrink_to_fit();
  n_grid = 0;
  for (auto &i: block) {
    n_grid += i.n_grid;
  } //compute the total grid number in this process
  for (integer blk = 0; blk < n_block; ++blk) {
    //read grid coordinates
    for (integer k = 0; k < block[blk].mz; ++k) {
      for (integer j = 0; j < block[blk].my; ++j) {
        for (integer i = 0; i < block[blk].mx; ++i) {
          grd >> block[blk].x(i, j, k);
        }
      }
    }
    for (integer k = 0; k < block[blk].mz; ++k) {
      for (integer j = 0; j < block[blk].my; ++j) {
        for (integer i = 0; i < block[blk].mx; ++i) {
          grd >> block[blk].y(i, j, k);
        }
      }
    }
    for (integer k = 0; k < block[blk].mz; ++k) {
      for (integer j = 0; j < block[blk].my; ++j) {
        for (integer i = 0; i < block[blk].mx; ++i) {
          grd >> block[blk].z(i, j, k);
        }
      }
    }
  }
  grd.close();
  fmt::print("Grid number in process {} is {}.\n", myid, n_grid);
  MPI_Barrier(MPI_COMM_WORLD);
  //Sum grid number in all processes to the root process
  MPI_Reduce(&n_grid, &n_grid_total, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Bcast(&n_grid_total, 1, MPI_INT, 0, MPI_COMM_WORLD); //Broadcast total grid number to all processes

  // Specify how many blocks there are on each process
  MPI_Allgather(&n_block, 1, MPI_INT, nblk, 1, MPI_INT, MPI_COMM_WORLD);
  auto *disp = new integer[n_proc];
  disp[0] = 0;
  n_block_total = nblk[0];
  for (int i = 1; i < n_proc; ++i) {
    disp[i] = disp[i - 1] + nblk[i - 1];
    n_block_total += nblk[i];
  }

  mx_blk = new integer[n_block_total];
  my_blk = new integer[n_block_total];
  mz_blk = new integer[n_block_total];
  auto *m_this = new integer[n_block * 3];
  integer *mx_this = m_this, *my_this = &mx_this[n_block], *mz_this = &my_this[n_block];
  for (int i = 0; i < n_block; ++i) {
    mx_this[i] = block[i].mx;
    my_this[i] = block[i].my;
    mz_this[i] = block[i].mz;
  }
  MPI_Allgatherv(mx_this, n_block, MPI_INT, mx_blk, nblk, disp, MPI_INT, MPI_COMM_WORLD);
  MPI_Allgatherv(my_this, n_block, MPI_INT, my_blk, nblk, disp, MPI_INT, MPI_COMM_WORLD);
  MPI_Allgatherv(mz_this, n_block, MPI_INT, mz_blk, nblk, disp, MPI_INT, MPI_COMM_WORLD);
  delete[]m_this;

//  if (myid == 0) {
//    fmt::print("There are {} blocks in all, the block number in each process is:\n", n_block_total);
//    for (int i = 0; i < n_proc; ++i) {
//      fmt::print("{}\n", n_block);
//    }
//    fmt::print("The dimension of the blocks are:\n");
//    for (int i = 0; i < n_block_total; ++i) {
//      fmt::print("Block {}:, mx = {}, my = {}, mz = {}\n", i, mx_blk[i], my_blk[i], mz_blk[i]);
//    }
//  }
}

void cfd::Mesh::read_boundary(integer myid/*, const integer ngg*/) {
  std::ifstream grd(fmt::format("./input_files/boundary_condition/boundary{:>4}.txt", myid), std::ios::in);
  std::string input{};
  for (integer blk = 0; blk < n_block; ++blk) {
    std::getline(grd, input);
    const integer bsz = std::stoi(input);
    integer i1 = 0, i2 = 0, i3 = 0, i4 = 0, i5 = 0, i6 = 0, i7 = 0;
    for (integer ii = 0; ii < bsz; ++ii) {
      std::getline(grd, input);
      std::istringstream line(input);
      line >> i1 >> i2 >> i3 >> i4 >> i5 >> i6 >> i7;
      block[blk].boundary.emplace_back(i1, i2, i3, i4, i5, i6, i7);
      block[blk].boundary[ii].register_boundary(ngg, dimension);
    }
    block[blk].boundary.shrink_to_fit();
  }
  grd.close();
  if (myid == 0) {
    fmt::print("Finish reading boundary conditions.\n");
  }
}

void cfd::Mesh::read_inner_interface(const integer myid/*, integer ngg*/) {
  std::ifstream grd(fmt::format("./input_files/boundary_condition/inner{:>4}.txt", myid), std::ios::in);
  std::string input{};
  // Read how many inner faces there are in each block and store them in @n_face.
  std::getline(grd, input);
  auto n_face = new integer[n_block];
  for (integer i = 0; i < n_block; ++i) {
    grd >> n_face[i];
  }
  std::getline(grd, input);
  std::getline(grd, input);
  for (integer blk = 0; blk < n_block; ++blk) {
    integer i1{0}, i2{0}, i3{0}, i4{0}, i5{0}, i6{0}, i7{0};
    integer j1{0}, j2{0}, j3{0}, j4{0}, j5{0}, j6{0}, j7{0};
    for (integer l = 0; l < n_face[blk]; ++l) {
      std::getline(grd, input);
      std::istringstream line(input);
      line >> i1 >> i2 >> i3 >> i4 >> i5 >> i6 >> i7;
      std::getline(grd, input);
      line.clear();
      line.str(input);
      line >> j1 >> j2 >> j3 >> j4 >> j5 >> j6 >> j7;
      block[blk].inner_face.emplace_back(i1, i2, i3, i4, i5, i6,
                                         j1, j2, j3, j4, j5, j6, j7);
      block[blk].inner_face[l].register_boundary(ngg, dimension);
    }
  }
  if (myid == 0) {
    fmt::print("Finish reading inner communication faces.\n");
  }
  grd.close();
  delete[]n_face;
}

void cfd::Mesh::read_parallel_interface(const integer myid/*, integer ngg*/) {
  std::ifstream grd(fmt::format("./input_files/boundary_condition/parallel{:>4}.txt", myid), std::ios::in);
  std::string input{};
  integer blk_num1{0}, tot_face{0};
  //Read block number of the process first to see if it matches the grid file information.
  grd >> blk_num1 >> tot_face;
  if (blk_num1 != n_block) {
    fmt::print("Read parallel error! In process {}, ", myid);
    fmt::print("n_block in grid file is {}, in parallel file is {}\n", n_block, blk_num1);
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  // Store the number of parallel faces that need to communicate among processes in @n_face.
  auto *n_face = new integer[n_block];
  for (integer i = 0; i < n_block; ++i) {
    grd >> n_face[i];
  } //Read the number of parallel communication faces of each block
  std::getline(grd, input);
  std::getline(grd, input);
  for (integer blk = 0; blk < n_block; ++blk) {
    integer i1{0}, i2{0}, i3{0}, i4{0}, i5{0}, i6{0}, i7{0}, flag_s{0}, flag_r{0};
    for (integer f = 0; f < n_face[blk]; ++f) {
      grd >> i1 >> i2 >> i3 >> i4 >> i5 >> i6 >> i7 >> flag_s >> flag_r;
      block[blk].parallel_face.emplace_back(i1, i2, i3, i4, i5, i6,
                                            i7, flag_s, flag_r);
      block[blk].parallel_face[f].register_boundary(dimension, ngg);
    }
  }
  delete[]n_face;
  if (myid == 0) fmt::print("Finish reading parallel communication faces.\n");
  grd.close();
}

void cfd::Mesh::scale(const real scale) {
  for (integer b = 0; b < n_block; ++b) {
    for (auto &gg: block[b].x) {
      gg *= scale;
    }
    for (auto &gg: block[b].y) {
      gg *= scale;
    }
    for (auto &gg: block[b].z) {
      gg *= scale;
    }
  }
}

void cfd::Mesh::init_ghost_grid(const integer myid, const bool parallel/*, const integer ngg*/) {
  //Add ghost grid except corners
  for (integer blk = 0; blk < n_block; ++blk) {
    auto &B = block[blk];
    const auto mx = B.mx, my = B.my, mz = B.mz;
    //add ghost grid in i direction
    for (integer k = 0; k < mz; ++k) {
      for (integer j = 0; j < my; ++j) {
        for (integer i = -1; i >= -ngg - 1; --i) {
          B.x(i, j, k) = 2 * B.x(0, j, k) - B.x(-i, j, k);
          B.y(i, j, k) = 2 * B.y(0, j, k) - B.y(-i, j, k);
          B.z(i, j, k) = 2 * B.z(0, j, k) - B.z(-i, j, k);
        }
        for (integer i = mx; i < mx + ngg + 1; ++i) {
          B.x(i, j, k) = 2 * B.x(mx - 1, j, k) - B.x(2 * (mx - 1) - i, j, k);
          B.y(i, j, k) = 2 * B.y(mx - 1, j, k) - B.y(2 * (mx - 1) - i, j, k);
          B.z(i, j, k) = 2 * B.z(mx - 1, j, k) - B.z(2 * (mx - 1) - i, j, k);
        }
      }
    }
    //add ghost grid in j direction
    for (integer k = 0; k < mz; ++k) {
      for (integer i = 0; i < mx; ++i) {
        for (integer j = -1; j >= -ngg - 1; --j) {
          B.x(i, j, k) = 2 * B.x(i, 0, k) - B.x(i, -j, k);
          B.y(i, j, k) = 2 * B.y(i, 0, k) - B.y(i, -j, k);
          B.z(i, j, k) = 2 * B.z(i, 0, k) - B.z(i, -j, k);
        }
        for (integer j = my; j < my + ngg + 1; ++j) {
          B.x(i, j, k) = 2 * B.x(i, my - 1, k) - B.x(i, 2 * (my - 1) - j, k);
          B.y(i, j, k) = 2 * B.y(i, my - 1, k) - B.y(i, 2 * (my - 1) - j, k);
          B.z(i, j, k) = 2 * B.z(i, my - 1, k) - B.z(i, 2 * (my - 1) - j, k);
        }
      }
    }
    //add ghost grid in k direction
    if (dimension == 3) {
      for (integer j = 0; j < my; ++j) {
        for (integer i = 0; i < mx; ++i) {
          for (integer k = -1; k >= -ngg - 1; --k) {
            B.x(i, j, k) = 2 * B.x(i, j, 0) - B.x(i, j, -k);
            B.y(i, j, k) = 2 * B.y(i, j, 0) - B.y(i, j, -k);
            B.z(i, j, k) = 2 * B.z(i, j, 0) - B.z(i, j, -k);
          }
          for (integer k = mz; k < mz + ngg + 1; ++k) {
            B.x(i, j, k) = 2 * B.x(i, j, mz - 1) - B.x(i, j, 2 * (mz - 1) - k);
            B.y(i, j, k) = 2 * B.y(i, j, mz - 1) - B.y(i, j, 2 * (mz - 1) - k);
            B.z(i, j, k) = 2 * B.z(i, j, mz - 1) - B.z(i, j, 2 * (mz - 1) - k);
          }
        }
      }
    } else {
      // dimension = 2
      for (integer j = 0; j < my; ++j) {
        for (integer i = 0; i < mx; ++i) {
          for (integer k = -1; k >= -ngg - 1; --k) {
            B.x(i, j, k) = B.x(i, j, 0);
            B.y(i, j, k) = B.y(i, j, 0);
            B.z(i, j, k) = B.z(i, j, 0) + k;
          }
          for (integer k = mz; k < mz + ngg + 1; ++k) {
            B.x(i, j, k) = B.x(i, j, mz - 1);
            B.y(i, j, k) = B.y(i, j, mz - 1);
            B.z(i, j, k) = B.z(i, j, mz - 1) + k - mz + 1;
          }
        }
      }
    }
  }
  //Assign the ghost grid of inner boundary by neighbor grid
  init_inner_ghost_grid();
  //If we use parallel computation, assign the ghost grid of parallel boundaries by MPI
  if (parallel) {
    init_parallel_ghost_grid(myid/*, ngg*/);
  }
  /*Assign ghost grids on the edges and at the corners by using parallelogram hypothesis*/
  for (integer blk = 0; blk < n_block; ++blk) {
    auto &B = block[blk];
    const auto mx = B.mx, my = B.my, mz = B.mz;
    //The whole ghost grid range in z direction are assigned, which is  [-ngg-1,mx+ngg]*[-ngg-1,my+ngg]*[0,mz-1]
    for (integer i = 1; i <= ngg + 1; ++i) {
      for (integer j = 1; j <= ngg + 1; ++j) {
        for (integer k = 0; k < mz; ++k) {
          const integer i1{-i}, j1{-j};
          const integer i2{mx - 1 + i}, j2{my - 1 + j};
          B.x(i1, j1, k) = B.x(i1, 0, k) + B.x(0, j1, k) - B.x(0, 0, k);
          B.y(i1, j1, k) = B.y(i1, 0, k) + B.y(0, j1, k) - B.y(0, 0, k);
          B.z(i1, j1, k) = B.z(i1, 0, k) + B.z(0, j1, k) - B.z(0, 0, k);

          B.x(i2, j2, k) = B.x(i2, my - 1, k) + B.x(mx - 1, j2, k) - B.x(mx - 1, my - 1, k);
          B.y(i2, j2, k) = B.y(i2, my - 1, k) + B.y(mx - 1, j2, k) - B.y(mx - 1, my - 1, k);
          B.z(i2, j2, k) = B.z(i2, my - 1, k) + B.z(mx - 1, j2, k) - B.z(mx - 1, my - 1, k);

          B.x(i1, j2, k) = B.x(i1, my - 1, k) + B.x(0, j2, k) - B.x(0, my - 1, k);
          B.y(i1, j2, k) = B.y(i1, my - 1, k) + B.y(0, j2, k) - B.y(0, my - 1, k);
          B.z(i1, j2, k) = B.z(i1, my - 1, k) + B.z(0, j2, k) - B.z(0, my - 1, k);

          B.x(i2, j1, k) = B.x(i2, 0, k) + B.x(mx - 1, j1, k) - B.x(mx - 1, 0, k);
          B.y(i2, j1, k) = B.y(i2, 0, k) + B.y(mx - 1, j1, k) - B.y(mx - 1, 0, k);
          B.z(i2, j1, k) = B.z(i2, 0, k) + B.z(mx - 1, j1, k) - B.z(mx - 1, 0, k);
        }
      }
    }
    //And in y direction. [-ngg-1,mx+ngg]*[0,my-1]*[-ngg-1,my+ngg]
    for (integer i = 1; i <= ngg + 1; ++i) {
      for (integer j = 0; j < my; ++j) {
        for (integer k = 1; k <= ngg + 1; ++k) {
          const integer i1{-i}, k1{-k};
          const integer i2{mx - 1 + i}, k2{mz - 1 + k};
          B.x(i1, j, k1) = B.x(i1, j, 0) + B.x(0, j, k1) - B.x(0, j, 0);
          B.y(i1, j, k1) = B.y(i1, j, 0) + B.y(0, j, k1) - B.y(0, j, 0);
          B.z(i1, j, k1) = B.z(i1, j, 0) + B.z(0, j, k1) - B.z(0, j, 0);

          B.x(i2, j, k2) = B.x(i2, j, mz - 1) + B.x(mx - 1, j, k2) - B.x(mx - 1, j, mz - 1);
          B.y(i2, j, k2) = B.y(i2, j, mz - 1) + B.y(mx - 1, j, k2) - B.y(mx - 1, j, mz - 1);
          B.z(i2, j, k2) = B.z(i2, j, mz - 1) + B.z(mx - 1, j, k2) - B.z(mx - 1, j, mz - 1);

          B.x(i1, j, k2) = B.x(i1, j, mz - 1) + B.x(0, j, k2) - B.x(0, j, mz - 1);
          B.y(i1, j, k2) = B.y(i1, j, mz - 1) + B.y(0, j, k2) - B.y(0, j, mz - 1);
          B.z(i1, j, k2) = B.z(i1, j, mz - 1) + B.z(0, j, k2) - B.z(0, j, mz - 1);

          B.x(i2, j, k1) = B.x(i2, j, 0) + B.x(mx - 1, j, k1) - B.x(mx - 1, j, 0);
          B.y(i2, j, k1) = B.y(i2, j, 0) + B.y(mx - 1, j, k1) - B.y(mx - 1, j, 0);
          B.z(i2, j, k1) = B.z(i2, j, 0) + B.z(mx - 1, j, k1) - B.z(mx - 1, j, 0);
        }
      }
    }
    //The rest are grids in x direction and at the corners, which is the range defined by
    //[-ngg-1,mx+ngg]*([-ngg-1,-1]|[my,my+ngg])*([-ngg-1,-1]|[mz,mz+ngg])
    for (integer i = -ngg - 1; i <= mx + ngg; ++i) {
      for (integer j = 1; j <= ngg + 1; ++j) {
        for (integer k = 1; k <= ngg + 1; ++k) {
          const integer j1{-j}, k1{-k};
          const integer j2{my - 1 + j}, k2{mz - 1 + k};
          B.x(i, j1, k1) = B.x(i, j1, 0) + B.x(i, 0, k1) - B.x(i, 0, 0);
          B.x(i, j2, k2) = B.x(i, j2, mz - 1) + B.x(i, my - 1, k2) - B.x(i, my - 1, mz - 1);
          B.x(i, j1, k2) = B.x(i, j1, mz - 1) + B.x(i, 0, k2) - B.x(i, 0, mz - 1);
          B.x(i, j2, k1) = B.x(i, j2, 0) + B.x(i, my - 1, k1) - B.x(i, my - 1, 0);
          B.y(i, j1, k1) = B.y(i, j1, 0) + B.y(i, 0, k1) - B.y(i, 0, 0);
          B.y(i, j2, k2) = B.y(i, j2, mz - 1) + B.y(i, my - 1, k2) - B.y(i, my - 1, mz - 1);
          B.y(i, j1, k2) = B.y(i, j1, mz - 1) + B.y(i, 0, k2) - B.y(i, 0, mz - 1);
          B.y(i, j2, k1) = B.y(i, j2, 0) + B.y(i, my - 1, k1) - B.y(i, my - 1, 0);
          B.z(i, j1, k1) = B.z(i, j1, 0) + B.z(i, 0, k1) - B.z(i, 0, 0);
          B.z(i, j2, k2) = B.z(i, j2, mz - 1) + B.z(i, my - 1, k2) - B.z(i, my - 1, mz - 1);
          B.z(i, j1, k2) = B.z(i, j1, mz - 1) + B.z(i, 0, k2) - B.z(i, 0, mz - 1);
          B.z(i, j2, k1) = B.z(i, j2, 0) + B.z(i, my - 1, k1) - B.z(i, my - 1, 0);
        }
      }
    }
  }
  MPI_Barrier(MPI_COMM_WORLD);
  if (myid == 0) {
    fmt::print("Finish creating ghost grids.\n");
  }
}

void cfd::Mesh::init_inner_ghost_grid() {
  for (integer blk = 0; blk < n_block; ++blk) {
    auto &B = block[blk];
    const auto in_face = static_cast<integer>(B.inner_face.size());
    for (integer f = 0; f < in_face; ++f) {
      const auto &fc = B.inner_face[f];
      auto &nB = block[fc.target_block]; //n for neighbour
      integer di[]{0, 0, 0};
      integer min_[]{fc.range_start[0], fc.range_start[1], fc.range_start[2]},
          max_[]{fc.range_end[0], fc.range_end[1], fc.range_end[2]},
          mint_[]{fc.target_start[0], fc.target_start[1], fc.target_start[2]};
      // The following condition is only for this ghost grid assignment, otherwise no need of it
      if (fc.direction < 0) {
        min_[fc.face] -= 1;
        max_[fc.face] -= 1;
        if (fc.target_direction > 0) {
          mint_[fc.target_face] -= 1;
        } else {
          mint_[fc.target_face] += 1;
        }
      } else {
        max_[fc.face] += 1;
        min_[fc.face] += 1;
        if (fc.target_direction > 0) {
          mint_[fc.target_face] -= 1;
        } else {
          mint_[fc.target_face] += 1;
        }
      }
      const auto stx = fc.target_loop_dir[0], sty = fc.target_loop_dir[1], stz = fc.target_loop_dir[2];
      const auto di1 = fc.loop_dir[0], dj1 = fc.loop_dir[1], dk1 = fc.loop_dir[2];
      for (integer i1 = min_[0]; di1 * (i1 - max_[0]) != 1; i1 += di1) {
        di[0] = i1 - min_[0];
        for (integer j1 = min_[1]; dj1 * (j1 - max_[1]) != 1; j1 += dj1) {
          di[1] = j1 - min_[1];
          for (integer k1 = min_[2]; dk1 * (k1 - max_[2]) != 1; k1 += dk1) {
            di[2] = k1 - min_[2];
            const integer i2 = mint_[0] + stx * di[fc.src_tar[0]];
            const integer j2 = mint_[1] + sty * di[fc.src_tar[1]];
            const integer k2 = mint_[2] + stz * di[fc.src_tar[2]];
            B.x(i1, j1, k1) = nB.x(i2, j2, k2);
            B.y(i1, j1, k1) = nB.y(i2, j2, k2);
            B.z(i1, j1, k1) = nB.z(i2, j2, k2);
          }
        }
      }
    }
  }
}

void cfd::Mesh::init_parallel_ghost_grid(const integer myid/*, const integer ngg*/) {
  //Add up to the total face number
  size_t total_face = 0;
  for (integer m = 0; m < n_block; ++m) {
    total_face += block[m].parallel_face.size();
  }

  //A 2-D array which is the cache used when using MPI to send/recv messages. The first dimension is the face index
  //while the second dimension is the coordinate of that face, 3 consecutive number represents one position.
  const auto temp_s = new real *[total_face], temp_r = new real *[total_face];

  //Added with iterate through faces and will equal to the total face number when the loop ends
  integer fc_num = 0;
  //Compute the array size of different faces and allocate them. Different for different faces.
  for (integer blk = 0; blk < n_block; ++blk) {
    auto &B = block[blk];
    const auto fc = static_cast<integer>(B.parallel_face.size());
    for (integer f = 0; f < fc; ++f) {
      const auto &face = B.parallel_face[f];
      //The length of the array is 3*(ngg+1)*${number of grid points of the face}
      //ngg+1 is the number of layers to communicate, 3 for 3 coordinates(x,y,z)
      integer extent[3]{std::abs(face.range_start[0] - face.range_end[0]) + 1,
                        std::abs(face.range_start[1] - face.range_end[1]) + 1,
                        std::abs(face.range_start[2] - face.range_end[2]) + 1};
      for (int i = 0; i < 3; ++i) {
        if (i == face.face) continue;
        extent[i] += 2 * ngg + 2;
      }
      const integer len = 3 * (ngg + 1) * extent[0] * extent[1] * extent[2];
//      const integer len = 3 * (ngg + 1) * (std::abs(face.range_start[0] - face.range_end[0]) + 1)
//                          * (std::abs(face.range_end[1] - face.range_start[1]) + 1)
//                          * (std::abs(face.range_end[2] - face.range_start[2]) + 1);
      temp_s[fc_num] = new real[len];
      temp_r[fc_num] = new real[len];
      ++fc_num;
    }
  }

  //Create array for MPI_ISEND/IRecv
  //MPI_REQUEST is an array representing whether the face sends/recvs successfully
  const auto s_request = new MPI_Request[total_face], r_request = new MPI_Request[total_face];
  const auto s_status = new MPI_Status[total_face], r_status = new MPI_Status[total_face];
  fc_num = 0;

  for (integer m = 0; m < n_block; ++m) {
    auto &B = block[m];
    const auto f_num = static_cast<integer>(B.parallel_face.size());
    for (integer f = 0; f < f_num; ++f) {
      //Iterate through the faces
      const auto &Fc = B.parallel_face[f];
      integer num = 0;
      integer min_[]{Fc.range_start[0], Fc.range_start[1], Fc.range_start[2]}, max_[]{Fc.range_end[0], Fc.range_end[1],
                                                                                      Fc.range_end[2]};
      min_[0] = Fc.range_start[Fc.loop_order[0]] - Fc.direction;
      max_[0] = Fc.range_end[Fc.loop_order[0]] - Fc.direction * (ngg + 1);
      min_[1] = Fc.range_start[Fc.loop_order[1]] - Fc.loop_dir[Fc.loop_order[1]] * (ngg + 1);
      max_[1] = Fc.range_end[Fc.loop_order[1]] + Fc.loop_dir[Fc.loop_order[1]] * (ngg + 1);
      min_[2] = Fc.range_start[Fc.loop_order[2]] - Fc.loop_dir[Fc.loop_order[2]] * (ngg + 1);
      max_[2] = Fc.range_end[Fc.loop_order[2]] + Fc.loop_dir[Fc.loop_order[2]] * (ngg + 1);
      const integer di1 = -Fc.direction, dj1 = Fc.loop_dir[1];
      integer dk1 = Fc.loop_dir[2];
      if (dimension == 2) {
        min_[2] = -3;
        max_[2] = 3;
        dk1 = 1;
      }
      integer send_ijk[]{0, 0, 0};
      for (send_ijk[Fc.loop_order[0]] = min_[0]; di1 * (send_ijk[Fc.loop_order[0]] - max_[0]) != 1; send_ijk[Fc.
          loop_order[0]] += di1) {
        for (send_ijk[Fc.loop_order[1]] = min_[1]; dj1 * (send_ijk[Fc.loop_order[1]] - max_[1]) != 1; send_ijk[
                                                                                                          Fc.loop_order[1]] += dj1) {
          for (send_ijk[Fc.loop_order[2]] = min_[2]; dk1 * (send_ijk[Fc.loop_order[2]] - max_[2]) != 1;
               send_ijk[Fc.loop_order[2]] += dk1) {
            const integer i1{send_ijk[0]}, j1{send_ijk[1]}, k1{send_ijk[2]};
            temp_s[fc_num][num] = B.x(i1, j1, k1);
            temp_s[fc_num][num + 1] = B.y(i1, j1, k1);
            temp_s[fc_num][num + 2] = B.z(i1, j1, k1);
            num += 3;
          }
        }
      }
      //Send and receive. Take care of the first address!
      MPI_Isend(&temp_s[fc_num][0], num, MPI_DOUBLE, Fc.target_process, Fc.flag_send, MPI_COMM_WORLD,
                &s_request[fc_num]);
      MPI_Irecv(&temp_r[fc_num][0], num, MPI_DOUBLE, Fc.target_process, Fc.flag_receive, MPI_COMM_WORLD,
                &r_request[fc_num]);
      ++fc_num;
    }
  }

  //Wait for all faces finishing communication
  MPI_Waitall(static_cast<integer>(total_face), s_request, s_status);
  MPI_Waitall(static_cast<integer>(total_face), r_request, r_status);
  MPI_Barrier(MPI_COMM_WORLD);

  //Assign the correct value got by MPI receive
  fc_num = 0;
  for (integer blk = 0; blk < n_block; ++blk) {
    auto &B = block[blk];
    const size_t f_num = B.parallel_face.size();
    for (size_t f = 0; f < f_num; ++f) {
      const auto &fc = B.parallel_face[f];
      integer min_[]{0, 0, 0}, max_[]{0, 0, 0};
      min_[0] = fc.range_start[fc.loop_order[0]] + fc.direction;
      max_[0] = fc.range_end[fc.loop_order[0]] + fc.direction * (ngg + 1);
      min_[1] = fc.range_start[fc.loop_order[1]] - fc.loop_dir[fc.loop_order[1]] * (ngg + 1);
      max_[1] = fc.range_end[fc.loop_order[1]] + fc.loop_dir[fc.loop_order[1]] * (ngg + 1);
      min_[2] = fc.range_start[fc.loop_order[2]] - fc.loop_dir[fc.loop_order[2]] * (ngg + 1);
      max_[2] = fc.range_end[fc.loop_order[2]] + fc.loop_dir[fc.loop_order[2]] * (ngg + 1);
      const integer di1 = fc.direction, dj1 = fc.loop_dir[1];
      integer dk1 = fc.loop_dir[2];
      integer n{0};
      if (dimension == 2) {
        min_[2] = -3;
        max_[2] = 3;
        dk1 = 1;
      }
      integer recv_ijk[]{0, 0, 0};
      // Because x/y/z are stored independently, these values should be read continuously.
      for (recv_ijk[fc.loop_order[0]] = min_[0]; di1 * (recv_ijk[fc.loop_order[0]] - max_[0]) != 1; recv_ijk[fc.
          loop_order[0]] += di1) {
        for (recv_ijk[fc.loop_order[1]] = min_[1]; dj1 * (recv_ijk[fc.loop_order[1]] - max_[1]) != 1; recv_ijk[
                                                                                                          fc.loop_order[1]] += dj1) {
          for (recv_ijk[fc.loop_order[2]] = min_[2]; dk1 * (recv_ijk[fc.loop_order[2]] - max_[2]) != 1;
               recv_ijk[fc.loop_order[2]] += dk1) {
            const integer i1{recv_ijk[0]}, j1{recv_ijk[1]}, k1{recv_ijk[2]};
            B.x(i1, j1, k1) = temp_r[fc_num][n];
            B.y(i1, j1, k1) = temp_r[fc_num][n + 1];
            B.z(i1, j1, k1) = temp_r[fc_num][n + 2];
            n += 3;
          }
        }
      }
      fc_num++;
    }
  }

  //Free dynamic allocated memory
  delete[]s_status;
  delete[]r_status;
  delete[]s_request;
  delete[]r_request;
  for (size_t i = 0; i < total_face; ++i) {
    delete[]temp_s[i];
    delete[]temp_r[i];
  }
  delete[]temp_s;
  delete[]temp_r;
}
