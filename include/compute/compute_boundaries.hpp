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

  specfem::enums::element::boundary_tag top =
      specfem::enums::element::boundary_tag::none;
  specfem::enums::element::boundary_tag bottom =
      specfem::enums::element::boundary_tag::none;
  specfem::enums::element::boundary_tag left =
      specfem::enums::element::boundary_tag::none;
  specfem::enums::element::boundary_tag right =
      specfem::enums::element::boundary_tag::none;
  specfem::enums::element::boundary_tag bottom_right =
      specfem::enums::element::boundary_tag::none;
  specfem::enums::element::boundary_tag bottom_left =
      specfem::enums::element::boundary_tag::none;
  specfem::enums::element::boundary_tag top_right =
      specfem::enums::element::boundary_tag::none;
  specfem::enums::element::boundary_tag top_left =
      specfem::enums::element::boundary_tag::none;

  KOKKOS_FUNCTION boundary_types() = default;

  void update_boundary_type(const specfem::enums::boundaries::type &type,
                            const specfem::enums::element::boundary_tag &tag);
};

KOKKOS_FUNCTION bool
is_on_boundary(const specfem::enums::element::boundary_tag &tag,
               const specfem::compute::access::boundary_types &type,
               const int &iz, const int &ix, const int &ngllz,
               const int &ngllx);
} // namespace access

struct acoustic_free_surface {

  acoustic_free_surface(
      const specfem::kokkos::HostView1d<int> kmato,
      const std::vector<specfem::material::material *> materials,
      const specfem::mesh::boundaries::absorbing_boundary &absorbing_boundaries,
      const specfem::mesh::boundaries::acoustic_free_surface
          &acoustic_free_surface);

  int nelements;
  specfem::kokkos::DeviceView1d<int> ispec;
  specfem::kokkos::HostMirror1d<int> h_ispec;
  specfem::kokkos::DeviceView1d<specfem::compute::access::boundary_types> type;
  specfem::kokkos::HostMirror1d<specfem::compute::access::boundary_types>
      h_type;
};

struct stacey_medium {

  stacey_medium() = default;

  stacey_medium(
      const specfem::enums::element::type medium,
      const specfem::kokkos::HostView1d<int> kmato,
      const std::vector<specfem::material::material *> materials,
      const specfem::mesh::boundaries::absorbing_boundary &absorbing_boundaries,
      const specfem::mesh::boundaries::acoustic_free_surface
          &acoustic_free_surface);

  int nelements;
  specfem::kokkos::DeviceView1d<int> ispec;
  specfem::kokkos::HostMirror1d<int> h_ispec;
  specfem::kokkos::DeviceView1d<specfem::compute::access::boundary_types> type;
  specfem::kokkos::HostMirror1d<specfem::compute::access::boundary_types>
      h_type;
};

struct stacey {
  stacey(
      const specfem::kokkos::HostView1d<int> kmato,
      const std::vector<specfem::material::material *> materials,
      const specfem::mesh::boundaries::absorbing_boundary &absorbing_boundaries,
      const specfem::mesh::boundaries::acoustic_free_surface
          &acoustic_free_surface);

  int nelements;
  specfem::compute::stacey_medium elastic;
  specfem::compute::stacey_medium acoustic;
};

struct composite_stacey_dirichlet {
  composite_stacey_dirichlet(
      const specfem::kokkos::HostView1d<int> kmato,
      const std::vector<specfem::material::material *> materials,
      const specfem::mesh::boundaries::absorbing_boundary &absorbing_boundaries,
      const specfem::mesh::boundaries::acoustic_free_surface
          &acoustic_free_surface);

  int nelements;
  specfem::kokkos::DeviceView1d<int> ispec;
  specfem::kokkos::HostMirror1d<int> h_ispec;
  specfem::kokkos::DeviceView1d<specfem::compute::access::boundary_types> type;
  specfem::kokkos::HostMirror1d<specfem::compute::access::boundary_types>
      h_type;
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
      : acoustic_free_surface(kmato, materials, absorbing_boundaries,
                              acoustic_free_surface),
        stacey(kmato, materials, absorbing_boundaries, acoustic_free_surface),
        composite_stacey_dirichlet(kmato, materials, absorbing_boundaries,
                                   acoustic_free_surface) {}

  specfem::compute::acoustic_free_surface acoustic_free_surface; ///< acoustic
                                                                 ///< free
                                                                 ///< surface
                                                                 ///< boundary

  specfem::compute::stacey stacey;
  specfem::compute::composite_stacey_dirichlet composite_stacey_dirichlet;
};
} // namespace compute
} // namespace specfem

#endif
