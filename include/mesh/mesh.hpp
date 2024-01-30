#ifndef _MESH_HPP
#define _MESH_HPP

// #include "IO/fortran/read_material_properties.hpp"
#include "IO/fortran/read_mesh_database.hpp"
#include "boundaries/boundaries.hpp"
#include "control_nodes/control_nodes.hpp"
#include "coupled_interfaces/coupled_interfaces.hpp"
#include "elements/elements.hpp"
#include "kokkos_abstractions.h"
#include "materials/interface.hpp"
// #include "material_indic/material_indic.hpp"
#include "mpi_interfaces/mpi_interfaces.hpp"
#include "properties/properties.hpp"
#include "specfem_mpi/interface.hpp"
#include "specfem_setup.hpp"
#include <Kokkos_Core.hpp>
#include <memory>

namespace specfem {

namespace mesh {
/**
 * @brief Mesh Interface
 *
 * The mesh is implemented as a C++ struct. The mesh struct defines all the
 * variables nacessary to populate structs within specfem::compute namespace
 *
 */
struct mesh {

  int npgeo; ///< Total number of spectral element control nodes
  int nspec; ///< Total number of spectral elements
  int nproc; ///< Total number of processors
  specfem::mesh::control_nodes control_nodes; ///< Defines control nodes

  //   specfem::mesh::material_ind material_ind; ///< Struct used to store
  //                                             ///< material information for
  //                                             ///< every spectral element

  specfem::mesh::boundaries::absorbing_boundary abs_boundary; ///< Struct used
                                                              ///< to store data
                                                              ///< required to
                                                              ///< implement
                                                              ///< absorbing
                                                              ///< boundary

  specfem::mesh::properties parameters; ///< Struct to store simulation launch
                                        ///< parameters

  specfem::mesh::coupled_interfaces coupled_interfaces; ///< Struct to store
                                                        ///< coupled interfaces

  specfem::mesh::boundaries::acoustic_free_surface
      acfree_surface; ///< Struct used to store data required to implement
                      ///< acoustic free surface

  specfem::mesh::boundaries::forcing_boundary
      acforcing_boundary; ///< Struct used to store data required to implement
                          ///< acoustic forcing boundary

  specfem::mesh::elements::tangential_elements tangential_nodes; ///< Defines
                                                                 ///< tangential
                                                                 ///< nodes

  specfem::mesh::elements::axial_elements axial_nodes; ///< Defines axial nodes
  specfem::mesh::materials materials; ///< Defines material properties

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

  /**
   * @brief User output
   *
   */
  std::string print() const;
};
} // namespace mesh
} // namespace specfem

#endif
