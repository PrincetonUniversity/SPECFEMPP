#pragma once

#include "boundaries/absorbing_boundary.hpp"
#include "boundaries/boundaries.hpp"
#include "boundaries/free_surface.hpp"
#include "coordinates/coordinates.hpp"
#include "coupled_interfaces/coupled_interfaces.hpp"
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

  // Struct to store the coupled interfaces
  specfem::mesh::coupled_interfaces<dimension> coupled_interfaces;

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

  /**
   * @brief Constructor
   *
   * @param parameters Struct to store simulation launch parameters
   * @param coordinates Struct to store coordinates
   * @param mapping Struct to store mapping information
   * @param xix_regular Regular xi-xi mapping
   * @param jacobian_regular Regular Jacobian
   * @param irregular_element_number Kokkos View of irregular elements
   * @param partial_derivatives Struct to store partial derivatives
   * @param elements_types Struct to store element types
   * @param mass_matrix Struct to store mass matrix
   * @param materials Struct to store material properties
   * @param boundaries Struct to store information at the boundaries
   * @param absorbing_boundary Struct to store absorbing boundaries
   * @param free_surface Struct to store free surface boundaries
   * @param coupled_interfaces Struct to store coupled interfaces
   *
   * @note This constructor is usually unused, and the mesh is constructed
   *       using the @ref specfem::IO::read_3d_mesh function.
   *
   * @see  specfem::IO::read_3d_mesh
   *
   */
  mesh(const specfem::mesh::parameters<dimension> &parameters,
       const specfem::mesh::coordinates<dimension> &coordinates,
       const specfem::mesh::mapping<dimension> &mapping,
       const type_real xix_regular, const type_real jacobian_regular,
       const View1D<int> irregular_element_number,
       const specfem::mesh::partial_derivatives<dimension> &partial_derivatives,
       const specfem::mesh::element_types<dimension> &elements_types,
       const specfem::mesh::mass_matrix<dimension> &mass_matrix,
       const specfem::mesh::materials<dimension> &materials,
       const specfem::mesh::boundaries<dimension> &boundaries,
       const specfem::mesh::absorbing_boundary<dimension> &absorbing_boundary,
       const specfem::mesh::free_surface<dimension> &free_surface,
       const specfem::mesh::coupled_interfaces<dimension> &coupled_interfaces)
      : parameters(parameters), coordinates(coordinates), mapping(mapping),
        xix_regular(xix_regular), jacobian_regular(jacobian_regular),
        irregular_element_number(irregular_element_number),
        partial_derivatives(partial_derivatives),
        elements_types(elements_types), mass_matrix(mass_matrix),
        materials(materials), boundaries(boundaries),
        absorbing_boundary(absorbing_boundary), free_surface(free_surface),
        coupled_interfaces(coupled_interfaces){};

  ///@} // Constructors

  /**
   * @brief Print basic parameters of the mesh if interested
   */
  std::string print() const;
};
} // namespace mesh
} // namespace specfem
