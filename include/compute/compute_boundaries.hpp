#ifndef _COMPUTE_BOUNDARIES_HPP
#define _COMPUTE_BOUNDARIES_HPP

#include "enumerations/specfem_enums.hpp"
#include "kokkos_abstractions.h"
#include "mesh/mesh.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace compute {

struct acoustic_free_surface {

  acoustic_free_surface(const specfem::mesh::boundaries::acoustic_free_surface
                            &acoustic_free_surface);

  int nelem_acoustic_surface;
  specfem::kokkos::DeviceView1d<int> ispec_acoustic_surface;
  specfem::kokkos::HostMirror1d<int> h_ispec_acoustic_surface;
  specfem::kokkos::DeviceView1d<specfem::enums::boundaries::type> type;
  specfem::kokkos::HostMirror1d<specfem::enums::boundaries::type> h_type;
};

struct boundaries {
  /**
   * @brief Construct a new boundaries object
   *
   * @param boundaries mesh boundaries object providing the necessary
   * information about boundaries within the mesh
   */
  boundaries(const specfem::mesh::boundaries::acoustic_free_surface
                 &acoustic_free_surface)
      : acoustic_free_surface(acoustic_free_surface) {}

  specfem::compute::acoustic_free_surface acoustic_free_surface; ///< acoustic
                                                                 ///< free
                                                                 ///< surface
                                                                 ///< boundary
};

namespace access {
KOKKOS_FUNCTION bool
is_on_boundary(const specfem::enums::boundaries::type &type, const int &iz,
               const int &ix, const int &ngllz, const int &ngllx);
}
} // namespace compute
} // namespace specfem

#endif
