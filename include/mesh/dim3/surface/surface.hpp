#pragma once

#include "enumerations/dimension.hpp"
#include "enumerations/medium.hpp"
#include "mesh/mesh_base.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace mesh {

/**
 * @brief Struct to store communication information between different MPI slices
 *
 * @tparam DimensionTag Dimension type
 */
template <> struct surface<specfem::dimension::type::dim3> {

  constexpr static auto dimension =
      specfem::dimension::type::dim3; ///< Dimension type

  int nfaces_surface; ///< Number of surface faces
  int nspec;          ///< Number of elements
  int nglob;          ///< Number of nodes

  Kokkos::View<bool *, Kokkos::HostSpace> ispec_is_surface; ///< Surface
                                                            ///< elements
  Kokkos::View<bool *, Kokkos::HostSpace> iglob_is_surface; ///< Surface nodes

  /**
   * @name Constructors
   *
   */
  ///@{
  /**
   * @brief Default constructor
   *
   */
  surface() {};

  /**
   * @brief construct a container that tagged surface elements.
   *
   */
  surface(const int nfaces_surface, const int nspec, const int nglob)
      : nfaces_surface(nfaces_surface), nspec(nspec), nglob(nglob),
        ispec_is_surface("ispec_is_surface", nspec),
        iglob_is_surface("iglob_is_surface", nglob) {};

  ///@}
};

} // namespace mesh
} // namespace specfem
