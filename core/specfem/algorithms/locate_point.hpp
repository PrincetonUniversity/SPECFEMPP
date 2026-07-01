#pragma once

#include "locate_point/locate_point_impl.hpp"
#include "specfem/assembly/mesh.hpp"
#include "specfem/mesh.hpp"
#include "specfem/mpi.hpp"
#include "specfem/point.hpp"
#include "specfem/setup.hpp"
#include "specfem/utilities/is_close.hpp"
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace specfem {
namespace algorithms {

/**
 * @brief Result of locating a batch of points across MPI partitions.
 *
 * @tparam DimensionTag Spatial dimension (dim2 or dim3)
 */
template <specfem::element::dimension_tag DimensionTag>
struct LocatePointResult {
  std::vector<specfem::point::local_coordinates<DimensionTag>>
      local; ///< Located local coordinates; valid (ispec >= 0) only on the
             ///< owning rank, ispec = -1 elsewhere
  std::vector<int> partition_index; ///< MPI rank owning each point (replicated
                                    ///< on every rank)
  std::vector<type_real> error;     ///< Cartesian target-to-found distance in
                                    ///< metres (replicated on every rank)
};

/**
 * @brief Locate a batch of points across MPI partitions with inside-preference.
 *
 * Each rank tries to locate every point in its local mesh partition.  The
 * best owning rank for each point is selected according to two-tier priority
 * (matching the reference Fortran SPECFEM3D-Globe strategy):
 *
 *   1. Prefer elements where the recovered local coordinates are all within
 *      [-1, 1]  (the point is inside the element). Among tied ranks that
 *      satisfy this condition, the one with the smallest Cartesian back-
 *      projection distance is chosen.
 *   2. Fall back to minimum Cartesian distance when no rank owns an inside
 *      element for a given point.
 *
 * The selection is performed with exactly two MPI_Allreduce calls regardless
 * of the number of points, preserving the O(1)-communication efficiency of
 * the original batch approach.
 *
 * @param coords  Global coordinates of the points to locate.
 * @param mesh    This rank's local mesh partition.
 * @return A @ref LocatePointResult whose `local[i]` is valid (ispec >= 0) only
 *         when `partition_index[i]` equals this rank's MPI rank (ispec = -1
 *         otherwise), and whose `error[i]` (the target-to-found distance) is
 *         replicated on every rank.
 * @throws std::runtime_error if any point cannot be located on any rank.
 */
template <specfem::element::dimension_tag DimensionTag>
LocatePointResult<DimensionTag> locate_point(
    const std::vector<specfem::point::global_coordinates<DimensionTag>> &coords,
    const specfem::assembly::mesh<DimensionTag> &mesh) {

  const int npoints = static_cast<int>(coords.size());
  const int myrank = specfem::MPI::get_rank();

  // OUTSIDE_PENALTY ensures that any inside element (priority = dist) beats
  // any outside element (priority = OUTSIDE_PENALTY + dist) when taking the
  // global MPI_MIN.  It is large enough to dominate realistic Cartesian
  // distances while leaving headroom to add a finite dist without overflow.
  constexpr type_real OUTSIDE_PENALTY =
      std::numeric_limits<type_real>::max() / 4;

  // Per-point local state -------------------------------------------------
  std::vector<specfem::point::local_coordinates<DimensionTag>> local_lcoords(
      npoints);
  for (auto &lc : local_lcoords)
    lc.ispec = -1;

  // priority[i] encoding:
  //   dist                  – point inside element  (smallest wins globally)
  //   OUTSIDE_PENALTY + dist – point outside element (fallback to min dist)
  //   max()                  – point not found on this rank
  std::vector<type_real> local_priority(npoints,
                                        std::numeric_limits<type_real>::max());

  for (int i = 0; i < npoints; ++i) {
    try {
      const auto lcoord =
          specfem::algorithms::locate_point_impl::locate_point(coords[i], mesh);
      const auto found_global =
          specfem::algorithms::locate_point_impl::locate_point(lcoord, mesh);
      const type_real dist = specfem::point::distance(coords[i], found_global);

      local_lcoords[i] = lcoord;
      local_priority[i] = lcoord.inside() ? dist : (OUTSIDE_PENALTY + dist);
    } catch (const std::exception &) {
      // Point not in this rank's partition – leave as invalid / max priority.
    }
  }

  // Step 1: allreduce(min) to obtain the global best priority per point.
  // -------
  std::vector<type_real> global_priority = local_priority;
  SPECFEM_MPI_SAFECALL(MPI_Allreduce(MPI_IN_PLACE, global_priority.data(),
                                     npoints, SPECFEM_MPI_TYPE_REAL, MPI_MIN,
                                     MPI_COMM_WORLD));

  // Step 2: each rank claims points whose local priority matches the global
  // minimum; allreduce(max) resolves ties by assigning to the highest rank
  // (matching legacy SPECFEM behaviour).
  //
  // MPI_MIN is a comparison-selection that returns one contributed value
  // bit-for-bit on homogeneous clusters (MPI Standard §6.9.2), so exact
  // equality is correct in the common case.  However, on heterogeneous
  // clusters MPI may perform FP format conversion during the reduction
  // (§4.1), which can alter the low-order bits.  A relative tolerance of one
  // machine-epsilon guards against this for future heterogeneous deployments.
  std::vector<int> partition_index_selected(npoints, -1);
  for (int i = 0; i < npoints; ++i) {
    if (specfem::utilities::is_close(local_priority[i], global_priority[i],
                                     std::numeric_limits<type_real>::epsilon(),
                                     type_real(0)))
      partition_index_selected[i] = myrank;
  }
  SPECFEM_MPI_SAFECALL(MPI_Allreduce(MPI_IN_PLACE,
                                     partition_index_selected.data(), npoints,
                                     MPI_INT, MPI_MAX, MPI_COMM_WORLD));

  // Sanity check: every point must have been claimed by at least one rank.
  // -----
  for (int i = 0; i < npoints; ++i) {
    if (partition_index_selected[i] < 0) {
      throw std::runtime_error("Point " + std::to_string(i) +
                               " could not be located in any MPI partition");
    }
  }

  // Build output: non-owning ranks get ispec = -1. ----------------------------
  std::vector<specfem::point::local_coordinates<DimensionTag>> result_coords(
      npoints);
  for (int i = 0; i < npoints; ++i) {
    if (partition_index_selected[i] == myrank) {
      result_coords[i] = local_lcoords[i];
    } else {
      result_coords[i].ispec = -1;
    }
  }

  // Recover the winning back-projection distance from the reduced priority.
  // Inside points encode priority == dist; outside points encode
  // OUTSIDE_PENALTY + dist. global_priority is replicated on every rank, so the
  // error is communicated identically to partition_index_selected.
  std::vector<type_real> error(npoints);
  for (int i = 0; i < npoints; ++i) {
    error[i] = global_priority[i] >= OUTSIDE_PENALTY
                   ? global_priority[i] - OUTSIDE_PENALTY
                   : global_priority[i];
  }

  return { result_coords, partition_index_selected, error };
}

/**
 * @brief Path used to project a point onto a surface.
 *
 * Only along_z (vertical) is currently implemented.
 * along_x, along_y, and ellipsoidal are reserved and will throw.
 */
enum class projection { along_x, along_y, along_z, ellipsoidal };

/**
 * @brief Project @p target onto @p surface along @p along, returning the
 * landing point.
 *
 * For @c along_z the result is @c {target.x,target.y,elevation}, where the
 * elevation is the free surface above @c (x,y), or 0 when @p surface is empty.
 * MPI-reduced to the rank owning the nearest face.
 *
 * @param mesh Assembled 3D mesh geometry
 * @param surface Faces defining the target surface
 * @param target Point to project (only components orthogonal to @p along used)
 * @param along Projection geometry; only @ref projection::along_z is
 * implemented
 * @return The surface point @p target projects to
 * @throws std::runtime_error for unimplemented geometries
 */
specfem::point::global_coordinates<specfem::element::dimension_tag::dim3>
project_onto_surface(
    const specfem::assembly::mesh<specfem::element::dimension_tag::dim3> &mesh,
    const specfem::mesh::acoustic_free_surface<
        specfem::element::dimension_tag::dim3> &surface,
    const specfem::point::global_coordinates<
        specfem::element::dimension_tag::dim3> &target,
    projection along = projection::along_z);

} // namespace algorithms
} // namespace specfem
