#pragma once

#include "boundaries/boundaries.hpp"
#include "control_nodes/control_nodes.hpp"
#include "coupled_interfaces/coupled_interfaces.hpp"
#include "elements/axial_elements.hpp"
#include "elements/tangential_elements.hpp"
#include "enumerations/interface.hpp"
#include "materials/materials.hpp"
#include "mesh/tags/tags.hpp"
#include "parameters/parameters.hpp"
#include "specfem_mpi/interface.hpp"
#include "specfem_setup.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {

namespace mesh {
/**
 * @brief Struct to store information about the mesh read from the database
 *
 */
struct mesh {

  int npgeo; ///< Total number of spectral element control nodes
  int nspec; ///< Total number of spectral elements
  int nproc; ///< Total number of processors
  specfem::mesh::control_nodes<specfem::dimension::type::dim2>
      control_nodes; ///< Defines control nodes

  specfem::mesh::parameters<specfem::dimension::type::dim2>
      parameters; ///< Struct to store simulation launch
                  ///< parameters (never used)

  specfem::mesh::coupled_interfaces coupled_interfaces; ///< Struct to store
                                                        ///< coupled interfaces

  specfem::mesh::boundaries boundaries; ///< Struct to store information at the
                                        ///< boundaries

  specfem::mesh::tags tags; ///< Struct to store tags for every spectral element

  specfem::mesh::elements::tangential_elements tangential_nodes; ///< Defines
                                                                 ///< tangential
                                                                 ///< nodes
                                                                 ///< (never
                                                                 ///< used)

  specfem::mesh::elements::axial_elements axial_nodes; ///< Defines axial nodes
                                                       ///< (never used)
  specfem::mesh::materials materials; ///< Defines material properties

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

  mesh(const int npgeo, const int nspec, const int nproc,
       const specfem::mesh::control_nodes<specfem::dimension::type::dim2>
           &control_nodes,
       const specfem::mesh::parameters<specfem::dimension::type::dim2>
           &parameters,
       const specfem::mesh::coupled_interfaces &coupled_interfaces,
       const specfem::mesh::boundaries &boundaries,
       const specfem::mesh::tags &tags,
       const specfem::mesh::elements::tangential_elements &tangential_nodes,
       const specfem::mesh::elements::axial_elements &axial_nodes,
       const specfem::mesh::materials &materials)
      : npgeo(npgeo), nspec(nspec), nproc(nproc), control_nodes(control_nodes),
        parameters(parameters), coupled_interfaces(coupled_interfaces),
        boundaries(boundaries), tags(tags), tangential_nodes(tangential_nodes),
        axial_nodes(axial_nodes), materials(materials){};
  ///@}

  std::string print() const;
};
} // namespace mesh
} // namespace specfem
