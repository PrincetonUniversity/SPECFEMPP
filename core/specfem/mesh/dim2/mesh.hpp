#pragma once

#include "adjacency_graph/adjacency_graph.hpp"
#include "boundaries/boundaries.hpp"
#include "control_nodes/control_nodes.hpp"
#include "elements/axial_elements.hpp"
#include "elements/tangential_elements.hpp"
#include "enumerations/dimension.hpp"
#include "enumerations/interface.hpp"
#include "materials/materials.hpp"
#include "materials/materials.tpp"
#include "parameters/parameters.hpp"
#include "specfem/mesh/mesh_base.hpp"

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

  specfem::mesh::parameters<dimension> parameters;

  specfem::mesh::boundaries<dimension> boundaries; ///< Struct to store
                                                   ///< information at the
                                                   ///< boundaries

  specfem::mesh::tags<dimension> tags; ///< Struct to store
                                       ///< tags for every
                                       ///< spectral
                                       ///< element

  specfem::mesh::elements::tangential_elements<dimension> tangential_nodes;

  specfem::mesh::elements::axial_elements<dimension> axial_nodes;

  specfem::mesh::materials<dimension> materials; ///< Defines material
                                                 ///< properties

  specfem::mesh::adjacency_graph<dimension> adjacency_graph;

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
   *
   * @code{.cpp}
   * // Example of how to use this constructor
   * specfem::mesh::mesh<dimension> mesh(
   *    npgeo, nspec, nproc, control_nodes, parameters, coupled_interfaces,
   *    boundaries, tags, tangential_nodes, axial_nodes, materials);
   * @endcode
   */
  mesh(const int npgeo, const int nspec, const int nproc,
       const specfem::mesh::control_nodes<dimension> &control_nodes,
       const specfem::mesh::parameters<dimension> &parameters,
       const specfem::mesh::boundaries<dimension> &boundaries,
       const specfem::mesh::tags<dimension> &tags,
       const specfem::mesh::elements::tangential_elements<dimension>
           &tangential_nodes,
       const specfem::mesh::elements::axial_elements<dimension> &axial_nodes,
       const specfem::mesh::materials<dimension> &materials)
      : npgeo(npgeo), nspec(nspec), nproc(nproc), control_nodes(control_nodes),
        parameters(parameters), boundaries(boundaries), tags(tags),
        tangential_nodes(tangential_nodes), axial_nodes(axial_nodes),
        materials(materials) {};
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

  /**
   * @brief Checks the consistency of the mesh data structure.
   *
   * This function verifies the internal consistency of the mesh, ensuring that
   * all mesh components (such as control nodes, boundaries, materials, etc.)
   * are correctly initialized and compatible with each other. It is intended
   * to catch configuration or initialization errors before running simulations.
   *
   * @throws std::runtime_error if any inconsistency is detected within the
   * mesh.
   *
   */
  void check_consistency() const;

  /**
   * @brief Setup and check coupled interfaces in the mesh
   *
   * This function checks whether the coupled interfaces are correctly defined
   * in the mesh data structure and are contistent with the adjacency graph.
   *
   * @note This function should be called after the mesh has been fully
   * initialized and all components (control nodes, materials, boundaries)
   * have been populated from MESHFEM2D database files.
   */
  void setup_coupled_interfaces(
      const std::set<std::pair<int, int> > &coupled_interfaces);
};
} // namespace mesh
} // namespace specfem
