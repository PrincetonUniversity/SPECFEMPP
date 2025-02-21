#pragma once

#include "boundaries/absorbing_boundary.hpp"
#include "boundaries/boundaries.hpp"
#include "boundaries/free_surface.hpp"
#include "coordinates/coordinates.hpp"
#include "element_types/element_types.hpp"
#include "mass_matrix/mass_matrix.hpp"
#include "materials/materials.hpp"
#include "mesh/dim3/mapping/mapping.hpp"
#include "mesh/mesh_base.hpp"
#include "parameters/parameters.hpp"
#include "partial_derivatives/partial_derivatives.hpp"
#include "specfem_mpi/interface.hpp"
#include "specfem_setup.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {

namespace mesh {

template <> struct mesh<specfem::dimension::type::dim3> {

  constexpr static auto dimension =
      specfem::dimension::type::dim3; ///< Dimension

  template <typename T> using View1D = Kokkos::View<T *, Kokkos::HostSpace>;

  // Struct to store all the mesh parameter
  specfem::mesh::parameters<dimension> parameters;

  // Struct to store all the coordinates
  specfem::mesh::coordinates<dimension> coordinates;

  // Struct to store the mapping information
  specfem::mesh::mapping<dimension> mapping;

  // Irregular elements Kokkos
  type_real xix_regular, jacobian_regular;
  View1D<int> irregular_element_number;

  // Struct to store the partial derivatives
  specfem::mesh::partial_derivatives<dimension> partial_derivatives;

  // Struct to store element_types
  specfem::mesh::element_types<dimension> elements_types;

  // Mass matrix
  specfem::mesh::mass_matrix<dimension> mass_matrix;

  // Material
  specfem::mesh::materials<dimension> materials;

  // Struct to store the boundaries
  specfem::mesh::boundaries<dimension> boundaries;

  // Struct to store the absorbing boundaries
  specfem::mesh::absorbing_boundary<dimension> absorbing_boundary;

  // Struct to store the free surface
  specfem::mesh::free_surface<dimension> free_surface;

  /**
   * @name Constructors
   *
   */
  ///@{
  /**
   * @brief Default mesh constructor
   *
   */
  mesh(){};

  std::string print() const;
};
} // namespace mesh
} // namespace specfem
