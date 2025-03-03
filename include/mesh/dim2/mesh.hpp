#pragma once

#include "boundaries/boundaries.hpp"
#include "control_nodes/control_nodes.hpp"
#include "coupled_interfaces/coupled_interfaces.hpp"
#include "elements/axial_elements.hpp"
#include "elements/tangential_elements.hpp"
#include "enumerations/dimension.hpp"
#include "enumerations/interface.hpp"
#include "materials/materials.hpp"
#include "materials/materials.tpp"
#include "mesh/mesh_base.hpp"
#include "parameters/parameters.hpp"
#include "specfem_mpi/interface.hpp"
#include "specfem_setup.hpp"
#include "tags/tags.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {

namespace mesh {

template <> struct mesh<specfem::dimension::type::dim2> {

  constexpr static auto dimension =
      specfem::dimension::type::dim2; ///< Dimension

  int npgeo; ///< Total number of spectral element control nodes
  int nspec; ///< Total number of spectral elements
  int nproc; ///< Total number of processors
  specfem::mesh::control_nodes<dimension> control_nodes; ///< Defines control
                                                         ///< nodes

  specfem::mesh::parameters<dimension> parameters; ///< Struct to store
                                                   ///< simulation launch
                                                   ///< parameters (never used)

  specfem::mesh::coupled_interfaces<dimension>
      coupled_interfaces; ///< Struct to store
                          ///< coupled interfaces

  specfem::mesh::boundaries<dimension> boundaries; ///< Struct to store
                                                   ///< information at the
                                                   ///< boundaries

  specfem::mesh::tags<dimension> tags; ///< Struct to store
                                       ///< tags for every
                                       ///< spectral
                                       ///< element

  specfem::mesh::elements::tangential_elements<dimension>
      tangential_nodes; ///< Defines
                        ///< tangential
                        ///< nodes
                        ///< (never
                        ///< used)

  specfem::mesh::elements::axial_elements<dimension> axial_nodes; ///< Defines
                                                                  ///< axial
                                                                  ///< nodes
                                                                  ///< (never
                                                                  ///< used)
  specfem::mesh::materials<dimension> materials; ///< Defines material
                                                 ///< properties

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
   * @brief Mesh constructor
   *
   * This constructor initializes the mesh struct with the given parameters
   *
   * @param npgeo Total number of spectral element control nodes
   * @param nspec Total number of spectral elements
   * @param nproc Total number of processors
   * @param control_nodes Struct to store control nodes
   * @param parameters Struct to store simulation launch parameters
   * @param coupled_interfaces Struct to store coupled interfaces
   * @param boundaries Struct to store information at the boundaries
   * @param tags Struct to store tags for every spectral element
   * @param tangential_nodes Struct to store tangential nodes
   * @param axial_nodes Struct to store axial nodes
   * @param materials Struct to store material properties
   *
   * @see ::specfem::mesh::control_nodes, ::specfem::mesh::parameters,
   *      ::specfem::mesh::coupled_interfaces, ::specfem::mesh::boundaries,
   *      ::specfem::mesh::tags, ::specfem::mesh::elements::tangential_elements,
   *      ::specfem::mesh::elements::axial_elements, ::specfem::mesh::materials
   *
   * @code{.cpp}
   * // Example of how to use this constructor
   * specfem::mesh::mesh<specfem::dimension::type::dim2> mesh(
   *    npgeo, nspec, nproc, control_nodes, parameters, coupled_interfaces,
   *    boundaries, tags, tangential_nodes, axial_nodes, materials);
   * @endcode
   */
  mesh(
      const int npgeo, const int nspec, const int nproc,
      const specfem::mesh::control_nodes<specfem::dimension::type::dim2>
          &control_nodes,
      const specfem::mesh::parameters<specfem::dimension::type::dim2>
          &parameters,
      const specfem::mesh::coupled_interfaces<specfem::dimension::type::dim2>
          &coupled_interfaces,
      const specfem::mesh::boundaries<specfem::dimension::type::dim2>
          &boundaries,
      const specfem::mesh::tags<specfem::dimension::type::dim2> &tags,
      const specfem::mesh::elements::tangential_elements<
          specfem::dimension::type::dim2> &tangential_nodes,
      const specfem::mesh::elements::axial_elements<
          specfem::dimension::type::dim2> &axial_nodes,
      const specfem::mesh::materials<specfem::dimension::type::dim2> &materials)
      : npgeo(npgeo), nspec(nspec), nproc(nproc), control_nodes(control_nodes),
        parameters(parameters), coupled_interfaces(coupled_interfaces),
        boundaries(boundaries), tags(tags), tangential_nodes(tangential_nodes),
        axial_nodes(axial_nodes), materials(materials){};
  ///@} // Constructors

  /**
   * @brief Print mesh information
   *
   * This function prints the mesh information
   *
   * @return std::string String containing the mesh information
   *
   * @code{.cpp}
   * // Example of how to use this function
   * std::string mesh_info = mesh.print();
   * @endcode
   */
  std::string print() const;
};
} // namespace mesh
} // namespace specfem
