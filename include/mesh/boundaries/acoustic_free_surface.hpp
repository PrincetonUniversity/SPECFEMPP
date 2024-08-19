#ifndef _ACOUSTIC_FREE_SURFACE_HPP
#define _ACOUSTIC_FREE_SURFACE_HPP

#include "enumerations/specfem_enums.hpp"
#include "kokkos_abstractions.h"
#include "specfem_mpi/interface.hpp"

namespace specfem {
namespace mesh {

struct acoustic_free_surface {
  acoustic_free_surface(){};
  acoustic_free_surface(const int nelem_acoustic_surface);
  acoustic_free_surface(std::ifstream &stream,
                        const int &nelem_acoustic_surface,
                        const specfem::kokkos::HostView2d<int> &knods,
                        const specfem::MPI::MPI *mpi);

  int nelem_acoustic_surface; ///< Number of elements on the acoustic free
                              ///< surface
  specfem::kokkos::HostView1d<int> ispec_acoustic_surface; ///< Number of
                                                           ///< elements on the
                                                           ///< acoustic free
                                                           ///< surface
  specfem::kokkos::HostView1d<specfem::enums::boundaries::type>
      type; ///< Type of the boundary
};
} // namespace mesh
} // namespace specfem

#endif
