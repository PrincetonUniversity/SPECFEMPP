#ifndef _MESH_PROPERTIES_HPP
#define _MESH_PROPERTIES_HPP

#include "specfem_mpi/interface.hpp"

namespace specfem {

namespace mesh {
/**
 * @brief Mesh properties
 *
 * Mesh properties stores meta-data used to allocate other structs within the
 * Mesh interface.
 */
struct properties {
  int numat;           ///< Total number of different materials
  int ngnod;           ///< Number of control nodes
  int nspec;           ///< Number of spectral elements
  int pointsdisp;      // Total number of points to display (Only used for
                       // visualization)
  int nelemabs;        ///< Number of elements on absorbing boundary
  int nelem_acforcing; ///< Number of elements on acoustic forcing boundary
  int nelem_acoustic_surface;  ///< Number of elements on acoustic surface
  int num_fluid_solid_edges;   ///< Number of solid-fluid edges
  int num_fluid_poro_edges;    ///< Number of fluid-poroelastic edges
  int num_solid_poro_edges;    ///< Number of solid-poroelastic edges
  int nnodes_tangential_curve; ///< Number of elements on tangential curve
  int nelem_on_the_axis;       ///< Number of axial elements
  bool plot_lowerleft_corner_only;

  /**
   * @brief Default constructor
   *
   */
  properties(){};
  /**
   * Constructor to read and assign values from fortran binary database file
   *
   * @param stream Stream object for fortran binary file buffered to properties
   * section
   * @param mpi Pointer to MPI object
   */
  properties(std::ifstream &stream, const specfem::MPI::MPI *mpi);
};
} // namespace mesh
} // namespace specfem

#endif
