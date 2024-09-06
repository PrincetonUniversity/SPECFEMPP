#pragma once

#include "boundaries/boundaries.hpp"
#include "control_nodes/control_nodes.hpp"
#include "coupled_interfaces/coupled_interfaces.hpp"
#include "elements/axial_elements.hpp"
#include "elements/tangential_elements.hpp"
#include "materials/materials.hpp"
#include "mesh/tags/tags.hpp"
#include "properties/properties.hpp"
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
  specfem::mesh::control_nodes control_nodes; ///< Defines control nodes

  specfem::mesh::properties parameters; ///< Struct to store simulation launch
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

  /**
   * @brief Construct mesh from a fortran binary database file
   *
   * @param filename Fortran binary database filename
   * @param mpi pointer to MPI object to manage communication
   */
  mesh(const std::string filename, const specfem::MPI::MPI *mpi);
  ///@}

  std::string print() const;
};
} // namespace mesh
} // namespace specfem
