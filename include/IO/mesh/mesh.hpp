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

namespace IO {

  /* @brief Construct a mesh object from a Fortran binary database file
   *
   * @param filename Fortran binary database filename
   * @param mpi pointer to MPI object to manage communication
   */
  specfem::mesh::mesh read_mesh(const std::string filename, const specfem::MPI::MPI *mpi);
  
} // namespace IO
} // namespace specfem
