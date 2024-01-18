#ifndef _MATERIAL_INDIC_HPP
#define _MATERIAL_INDIC_HPP

#include "kokkos_abstractions.h"
#include "specfem_mpi/interface.hpp"

namespace specfem {
namespace mesh {
struct material_ind {

  /**
   * @brief Defines the type of PML
   * @code
   * if (region_CPML(ispec) == 0) means ispec is not a PML element
   * if (region_CPML(ispec) == 1) means ispec is a X PML element
   * if (region_CPML(ispec) == 2) means ispec is a Z PML element
   * if (region_CPML(ispec) == 3) means ispec is a XZ PML element
   * @endcode
   */
  specfem::kokkos::HostView1d<int> region_CPML;

  specfem::kokkos::HostView1d<int> kmato; ///< Defines material specification
                                          ///< number

  /**
   * @brief Default constructor
   *
   */
  material_ind(){};
  /**
   * @brief Constructor used to allocate views
   *
   * @param nspec Number of spectral elements
   * @param ngnod Number of control nodes per spectral element
   */
  material_ind(const int nspec, const int ngnod);
  /**
   * @brief Constructor used to allocate and assign views from fortran database
   * file
   *
   * @param stream Stream object for fortran binary file buffered to material
   * definition section
   * @param ngnod Number of control nodes per spectral element
   * @param nspec Number of spectral elements
   * @param numat Total number of different materials
   * @param mpi Pointer to a MPI object
   */
  material_ind(std::ifstream &stream, const int ngnod, const int nspec,
               const int numat, const specfem::kokkos::HostView2d<int> knods,
               const specfem::MPI::MPI *mpi);
};
} // namespace mesh
} // namespace specfem

#endif
