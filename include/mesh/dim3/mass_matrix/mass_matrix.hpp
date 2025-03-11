#pragma once

#include "enumerations/dimension.hpp"
#include "mesh/mesh_base.hpp"
#include "specfem_setup.hpp"

namespace specfem {
namespace mesh {

/**
 * @brief Struct to store mass matrix for a 3D mesh
 *
 */
template <> struct mass_matrix<specfem::dimension::type::dim3> {

  constexpr static auto dimension = specfem::dimension::type::dim3;

  int nglob;           ///< Number of global nodes
  bool is_acoustic;    ///< Is acoustic
  bool is_elastic;     ///< Is elastic
  bool is_poroelastic; ///< Is poroelastic
  bool has_ocean_load; ///< Has ocean load

  template <typename T> using View1D = Kokkos::View<T *, Kokkos::HostSpace>;

  View1D<type_real> elastic;           ///< Elastic mass matrix
  View1D<type_real> acoustic;          ///< Acoustic mass matrix
  View1D<type_real> ocean_load;        ///< Ocean load mass matrix
  View1D<type_real> solid_poroelastic; ///< Solid poroelastic mass matrix
  View1D<type_real> fluid_poroelastic; ///< Fluid poroelastic mass matrix

  /**
   * @name Constructors
   *
   */
  ///@{
  /**
   * @brief Default constructor initializing an empty struct
   *
   */
  mass_matrix() = default;

  /**
   * @brief Construct a new mass matrix object
   *
   * @param nglob Number of global nodes
   * @param is_acoustic Is acoustic
   * @param is_elastic Is elastic
   * @param is_poroelastic Is poroelastic
   * @param has_ocean_load Has ocean load
   *
   * @code{.cpp}
   * // Example of how to use this constructor
   * int nglob = 10;
   * bool is_acoustic = true;
   * bool is_elastic = false;
   * bool is_poroelastic = false;
   * bool has_ocean_load = false;
   * specfem::mesh::mass_matrix<specfem::dimension::type::dim3>
   *   mass_matrix(nglob, is_acoustic, is_elastic, is_poroelastic,
   * has_ocean_load);
   * @endcode
   */
  mass_matrix(const int nglob, const bool is_acoustic, const bool is_elastic,
              const bool is_poroelastic, const bool has_ocean_load)
      : nglob(nglob), is_acoustic(is_acoustic), is_elastic(is_elastic),
        is_poroelastic(is_poroelastic), has_ocean_load(has_ocean_load) {
    // Only allocate the mass matrix if the simulation includes the respective
    // physics.
    if (is_acoustic) {
      acoustic =
          View1D<type_real>("specfem::mesh::mass_matrix::acoustic", nglob);
    }
    if (is_elastic) {
      elastic = View1D<type_real>("specfem::mesh::mass_matrix::elastic", nglob);
      if (has_ocean_load) {
        ocean_load =
            View1D<type_real>("specfem::mesh::mass_matrix::ocean_load", nglob);
      }
    }
    if (is_poroelastic) {
      solid_poroelastic = View1D<type_real>(
          "specfem::mesh::mass_matrix::solid_poroelastic", nglob);
      fluid_poroelastic = View1D<type_real>(
          "specfem::mesh::mass_matrix::fluid_poroelastic", nglob);
    }
  };
  ///@}
};

} // namespace mesh
} // namespace specfem
