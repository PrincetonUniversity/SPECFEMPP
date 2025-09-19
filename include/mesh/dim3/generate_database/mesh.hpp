#pragma once

#include "adjacency/adjacency.hpp"
#include "boundaries/absorbing_boundary.hpp"
#include "boundaries/acoustic_free_surface.hpp"
#include "boundaries/boundaries.hpp"
#include "coloring/coloring.hpp"
#include "control_nodes/control_nodes.hpp"
#include "coordinates/coordinates.hpp"
#include "coupled_interfaces/coupled_interfaces.hpp"
#include "element_types/element_types.hpp"
#include "inner_outer/inner_outer.hpp"
#include "jacobian_matrix/jacobian_matrix.hpp"
#include "mass_matrix/mass_matrix.hpp"
#include "materials/materials.hpp"
#include "mesh/dim3/generate_database/element_types/element_types.hpp"
#include "mesh/dim3/generate_database/mapping/mapping.hpp"
#include "mesh/mesh_base.hpp"
#include "mpi/mpi.hpp"
#include "parameters/parameters.hpp"
#include "specfem_mpi/interface.hpp"
#include "specfem_setup.hpp"
#include "surface/surface.hpp"
#include "tags/tags.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {

namespace mesh {

/**
 * @brief Struct to store a 3D mesh
 *
 */
template <> struct mesh<specfem::dimension::type::dim3> {

  constexpr static auto dimension =
      specfem::dimension::type::dim3; ///< Dimension

  template <typename T> using View1D = Kokkos::View<T *, Kokkos::HostSpace>;

  // Struct to store all the mesh parameter
  specfem::mesh::parameters<dimension> parameters; ///< Parameters

  // Struct to store all the coordinates
  specfem::mesh::coordinates<dimension> coordinates; ///< Coordinates

  // Struct to store the mapping information
  specfem::mesh::mapping<dimension> mapping; ///< Mapping

  // Irregular elements Kokkos
  type_real xix_regular, jacobian_regular; ///< Regular xi-xi mapping
  View1D<int> irregular_element_number;    ///< Irregular elements

  // Struct to store the Jacobian matrix
  specfem::mesh::jacobian_matrix<dimension> jacobian_matrix; ///< Partial
                                                             ///< derivatives

  // Struct to store element_types
  specfem::mesh::element_types<dimension> element_types; ///< Element types

  // Mass matrix
  specfem::mesh::mass_matrix<dimension> mass_matrix; ///< Mass matrix

  // Material
  specfem::mesh::materials<dimension> materials; ///< Materials

  // Struct to store the boundaries
  specfem::mesh::boundaries<dimension> boundaries; ///< Boundaries

  specfem::mesh::tags<dimension> tags; ///< Struct to store
                                       ///< tags for every
                                       ///< spectral
                                       ///< element

  // Struct to store the coupled interfaces
  specfem::mesh::coupled_interfaces<dimension>
      coupled_interfaces; ///< Coupled
                          ///< interfaces

  // MPI information
  specfem::mesh::mpi<dimension> mpi; ///< MPI interfaces

  // Inner outer elements
  specfem::mesh::inner_outer<dimension> inner_outer; ///< Inner outer elements

  // Coloring
  specfem::mesh::coloring<dimension> coloring; ///< Coloring

  // Surface
  specfem::mesh::surface<dimension> surface; ///< Surface

  // Adjacency
  specfem::mesh::adjacency<dimension> adjacency; ///< Adjacency

  // Control nodes
  specfem::mesh::control_nodes<dimension> control_nodes; ///< Control nodes

  /**
   * @name Constructors
   *
   */
  ///@{
  /**
   * @brief Default mesh constructor
   *
   */
  mesh() {};

  /**
   * @brief Constructor
   *
   * @param parameters Struct to store simulation launch parameters
   * @param coordinates Struct to store coordinates
   * @param mapping Struct to store mapping information
   * @param xix_regular Regular xi-xi mapping
   * @param jacobian_regular Regular Jacobian
   * @param irregular_element_number Kokkos View of irregular elements
   * @param jacobian_matrix Struct to store Jacobian matrix
   * @param element_types Struct to store element types
   * @param mass_matrix Struct to store mass matrix
   * @param materials Struct to store material properties
   * @param boundaries Struct to store information at the boundaries
   * @param coupled_interfaces Struct to store coupled interfaces
   * @param mpi Struct to store MPI information
   * @param inner_outer Struct to store inner outer elements
   * @param coloring Struct to store coloring information
   * @param adjacency Struct to store element adjacency information
   *
   *
   * @note This constructor is usually unused, and the mesh is constructed
   *       using the @ref specfem::io::read_3d_mesh function.
   *
   * @see  specfem::io::read_3d_mesh
   *
   */
  mesh(const specfem::mesh::parameters<dimension> &parameters,
       const specfem::mesh::coordinates<dimension> &coordinates,
       const specfem::mesh::mapping<dimension> &mapping,
       const type_real xix_regular, const type_real jacobian_regular,
       const View1D<int> irregular_element_number,
       const specfem::mesh::jacobian_matrix<dimension> &jacobian_matrix,
       const specfem::mesh::element_types<dimension> &element_types,
       const specfem::mesh::mass_matrix<dimension> &mass_matrix,
       const specfem::mesh::materials<dimension> &materials,
       const specfem::mesh::boundaries<dimension> &boundaries,
       const specfem::mesh::tags<dimension> &tags,
       const specfem::mesh::coupled_interfaces<dimension> &coupled_interfaces,
       const specfem::mesh::mpi<dimension> &mpi,
       const specfem::mesh::inner_outer<dimension> &inner_outer,
       const specfem::mesh::coloring<dimension> &coloring,
       const specfem::mesh::surface<dimension> &surface,
       const specfem::mesh::adjacency<dimension> &adjacency)
      : parameters(parameters), coordinates(coordinates), mapping(mapping),
        xix_regular(xix_regular), jacobian_regular(jacobian_regular),
        irregular_element_number(irregular_element_number),
        jacobian_matrix(jacobian_matrix), element_types(element_types),
        mass_matrix(mass_matrix), materials(materials), boundaries(boundaries),
        tags(tags), coupled_interfaces(coupled_interfaces), mpi(mpi),
        inner_outer(inner_outer), coloring(coloring), surface(surface),
        adjacency(adjacency) {};

  ///@} // Constructors

  /**
   * @brief Print basic parameters of the mesh if interested
   */
  std::string print() const;
};
} // namespace mesh
} // namespace specfem
