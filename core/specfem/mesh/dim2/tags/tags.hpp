#pragma once

#include "enumerations/boundary.hpp"
#include "enumerations/medium.hpp"
#include "specfem/mesh/dim2/boundaries/boundaries.hpp"
#include "specfem/mesh/dim2/materials/materials.hpp"
#include "specfem/mesh/impl/tags_container.hpp"
#include "specfem/mesh/mesh_base.hpp"

namespace specfem {
namespace mesh {

template <> struct tags<specfem::dimension::type::dim2> {

  constexpr static auto dimension =
      specfem::dimension::type::dim2; ///< Dimension

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
  tags() = default;

  /**
   * @brief Construct tags from mesh data
   *
   * @param materials Material properties
   * @param boundaries Boundary information
   */
  tags(
      const specfem::mesh::materials<specfem::dimension::type::dim2> &materials,
      const specfem::mesh::boundaries<specfem::dimension::type::dim2>
          &boundaries);
  ///@}
};
} // namespace mesh
} // namespace specfem
