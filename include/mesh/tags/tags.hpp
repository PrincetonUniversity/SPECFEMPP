#pragma once

#include "enumerations/boundary.hpp"
#include "enumerations/medium.hpp"
#include "mesh/boundaries/boundaries.hpp"
#include "mesh/materials/materials.hpp"
#include "tags_container.hpp"

namespace specfem {
namespace mesh {
/**
 * @brief Struct to store tags for every spectral element
 *
 */
template <specfem::dimension::type DimensionType> struct tags;

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
  tags(const specfem::mesh::materials &materials,
       const specfem::mesh::boundaries &boundaries);
  ///@}
};
} // namespace mesh
} // namespace specfem
