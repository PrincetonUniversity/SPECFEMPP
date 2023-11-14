#ifndef _COMPUTE_BOUNDARIES_HPP
#define _COMPUTE_BOUNDARIES_HPP

#include "enumerations/specfem_enums.hpp"
#include "kokkos_abstractions.h"
#include "material/interface.hpp"
#include "mesh/mesh.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace compute {

namespace access {

/**
 * @brief Struct to save boundary types for each element
 *
 */
struct boundary_types {

  bool top = false;
  bool bottom = false;
  bool left = false;
  bool right = false;
  bool bottom_right = false;
  bool bottom_left = false;
  bool top_right = false;
  bool top_left = false;

  KOKKOS_FUNCTION boundary_types() = default;

  void update_boundary_type(const specfem::enums::boundaries::type &type);
};

KOKKOS_FUNCTION bool
is_on_boundary(const specfem::compute::access::boundary_types &type,
               const int &iz, const int &ix, const int &ngllz,
               const int &ngllx);
} // namespace access

struct acoustic_free_surface {

  acoustic_free_surface(
      const specfem::kokkos::HostView1d<int> kmato,
      const std::vector<specfem::material::material *> materials,
      const specfem::mesh::boundaries::acoustic_free_surface
          &acoustic_free_surface);

  int nelem_acoustic_surface;
  specfem::kokkos::DeviceView1d<int> ispec_acoustic_surface;
  specfem::kokkos::HostMirror1d<int> h_ispec_acoustic_surface;
  specfem::kokkos::DeviceView1d<specfem::compute::access::boundary_types> type;
  specfem::kokkos::HostMirror1d<specfem::compute::access::boundary_types>
      h_type;
};

struct stacey_medium {

  stacey_medium() = default;

  stacey_medium(const specfem::enums::element::type medium,
                const specfem::kokkos::HostView1d<int> kmato,
                const std::vector<specfem::material::material *> materials,
                const specfem::mesh::boundaries::absorbing_boundary
                    &absorbing_boundaries);

  int nelements;
  specfem::kokkos::DeviceView1d<int> ispec;
  specfem::kokkos::HostMirror1d<int> h_ispec;
  specfem::kokkos::DeviceView1d<specfem::compute::access::boundary_types> type;
  specfem::kokkos::HostMirror1d<specfem::compute::access::boundary_types>
      h_type;
};

struct stacey {

  stacey(const specfem::kokkos::HostView1d<int> kmato,
         const std::vector<specfem::material::material *> materials,
         const specfem::mesh::boundaries::absorbing_boundary
             &absorbing_boundaries);

  int nelements;
  specfem::compute::stacey_medium elastic;
  specfem::compute::stacey_medium acoustic;
};

struct boundaries {
  /**
   * @brief Construct a new boundaries object
   *
   * @param boundaries mesh boundaries object providing the necessary
   * information about boundaries within the mesh
   */
  boundaries(
      const specfem::kokkos::HostView1d<int> kmato,
      const std::vector<specfem::material::material *> materials,
      const specfem::mesh::boundaries::acoustic_free_surface
          &acoustic_free_surface,
      const specfem::mesh::boundaries::absorbing_boundary &absorbing_boundaries)
      : acoustic_free_surface(kmato, materials, acoustic_free_surface),
        stacey(kmato, materials, absorbing_boundaries) {}

  specfem::compute::acoustic_free_surface acoustic_free_surface; ///< acoustic
                                                                 ///< free
                                                                 ///< surface
                                                                 ///< boundary

  specfem::compute::stacey stacey;
};
} // namespace compute
} // namespace specfem

#endif
