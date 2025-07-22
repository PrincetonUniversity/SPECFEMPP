#pragma once

#include "enumerations/interface.hpp"
#include "mesh/mesh.hpp"
#include "specfem/data_access.hpp"
#include <Kokkos_Core.hpp>

namespace specfem::assembly {

template <>
struct jacobian_matrix<specfem::dimension::type::dim3>
    : public specfem::data_access::Container<
          specfem::data_access::ContainerType::domain,
          specfem::data_access::DataClassType::jacobian_matrix,
          specfem::dimension::type::dim3> {
  using base_type = specfem::data_access::Container<
      specfem::data_access::ContainerType::domain,
      specfem::data_access::DataClassType::jacobian_matrix,
      specfem::dimension::type::dim3>;
  using view_type = typename base_type::scalar_type<
      type_real, Kokkos::DefaultExecutionSpace::memory_space>;

  int nspec; ///< Number of spectral elements
  int ngllx; ///< Number of GLL points in x
  int nglly; ///< Number of GLL points in y
  int ngllz; ///< Number of GLL points in z

  view_type xix;      ///< @xix
  view_type xiy;      ///< @xiy
  view_type xiz;      ///< @xiz
  view_type etax;     ///< @etax
  view_type etay;     ///< @etay
  view_type etaz;     ///< @etaz
  view_type gammax;   ///< @gammax
  view_type gammay;   ///< @gammay
  view_type gammaz;   ///< @gammaz
  view_type jacobian; ///< Jacobian

  view_type::HostMirror h_xix;      ///< Host mirror of @xix
  view_type::HostMirror h_xiy;      ///< Host mirror of @xiy
  view_type::HostMirror h_xiz;      ///< Host mirror of @xiz
  view_type::HostMirror h_etax;     ///< Host mirror of @etax
  view_type::HostMirror h_etay;     ///< Host mirror of @etay
  view_type::HostMirror h_etaz;     ///< Host mirror of @etaz
  view_type::HostMirror h_gammax;   ///< Host mirror of @gammax
  view_type::HostMirror h_gammay;   ///< Host mirror of @gamm
  view_type::HostMirror h_gammaz;   ///< Host mirror of @gammaz
  view_type::HostMirror h_jacobian; ///< Host mirror of Jacobian

  jacobian_matrix() = default;

  jacobian_matrix(
      const specfem::mesh::jacobian_matrix<dimension_tag> &mesh_jacobian);

  void sync_views();

  std::tuple<bool, Kokkos::View<bool *, Kokkos::DefaultHostExecutionSpace> >
  check_small_jacobian() const;
};

} // namespace specfem::assembly
