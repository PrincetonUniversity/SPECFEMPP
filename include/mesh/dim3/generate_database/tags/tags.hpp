#pragma once

#include "enumerations/boundary.hpp"
#include "enumerations/medium.hpp"
#include "mesh/dim3/generate_database/boundaries/boundaries.hpp"
#include "mesh/dim3/generate_database/materials/materials.hpp"
#include "mesh/impl/tags_container.hpp"
#include "mesh/mesh_base.hpp"

namespace specfem {
namespace mesh {

template <> struct tags<specfem::dimension::type::dim3> {

  constexpr static auto dimension =
      specfem::dimension::type::dim3; ///< Dimension

  int nspec; ///< Total number of spectral elements
  Kokkos::View<specfem::mesh::impl::tags_container *, Kokkos::HostSpace>
      tags_container; ///< Tags container

  /**
   * @name Constructors
   */
  ///@{
  /**
   * @brief Contrust tags object
   *
   */
  tags(const int nspec) : tags_container("specfem::mesh::tags::tags", nspec) {}

  /**
   * @brief Default constructor
   *
   */

  /**
   * @brief Construct tags from mesh data
   *
   * @param materials Material properties
   * @param boundaries Boundary information
   */
  tags() = default;

  /**
   * @brief Construct tags from mesh data
   *
   * @param materials Material properties
   * @param boundaries Boundary information
   */
  tags(const specfem::mesh::element_types<specfem::dimension::type::dim3>
           &element_types,
       const specfem::mesh::boundaries<specfem::dimension::type::dim3>
           &boundaries,
       const specfem::mesh::parameters<specfem::dimension::type::dim3>
           &parameters);
  ///@}
};
} // namespace mesh
} // namespace specfem
