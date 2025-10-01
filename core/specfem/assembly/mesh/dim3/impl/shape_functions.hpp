#pragma once

#include "enumerations/interface.hpp"
#include <Kokkos_Core.hpp>

namespace specfem::assembly::mesh_impl {
/**
 * @brief Shape function and their derivatives for every control node within the
 * mesh
 *
 */
template <> struct shape_functions<specfem::dimension::type::dim3> {
private:
  constexpr static int ndim = 3; ///< Number of dimensions
public:
  constexpr static auto dimension_tag =
      specfem::dimension::type::dim3; ///< Dimension
  int ngllz; ///< Number of quadrature points in z dimension
  int nglly; ///< Number of quadrature points in y dimension
  int ngllx; ///< Number of quadrature points in x dimension
  int ngnod; ///< Number of control nodes

  using ShapeFunctionViewType =
      Kokkos::View<type_real ****, Kokkos::LayoutRight,
                   Kokkos::DefaultExecutionSpace>;

  using DShapeFunctionViewType =
      Kokkos::View<type_real *****, Kokkos::LayoutRight,
                   Kokkos::DefaultExecutionSpace>;

  shape_functions(const int &ngllz, const int &nglly, const int &ngllx,
                  const int &ngnod)
      : ngllz(ngllz), nglly(nglly), ngllx(ngllx), ngnod(ngnod),
        shape3D("specfem::assembly::shape_functions::shape2D", ngllz, nglly,
                ngllx, ngnod),
        dshape3D("specfem::assembly::shape_functions::dshape2D", ngllz, nglly,
                 ngllx, ndim, ngnod),
        h_shape3D(Kokkos::create_mirror_view(shape3D)),
        h_dshape3D(Kokkos::create_mirror_view(dshape3D)) {}

  shape_functions(
      const Kokkos::View<type_real *, Kokkos::DefaultHostExecutionSpace> xi,
      const Kokkos::View<type_real *, Kokkos::DefaultHostExecutionSpace> eta,
      const Kokkos::View<type_real *, Kokkos::DefaultHostExecutionSpace> zeta,
      const int &ngll, const int &ngnod);

  shape_functions() = default;

  ShapeFunctionViewType shape3D;                 ///< Shape functions
  DShapeFunctionViewType dshape3D;               ///< Shape function
                                                 ///< derivatives
  ShapeFunctionViewType::HostMirror h_shape3D;   ///< Shape functions
  DShapeFunctionViewType::HostMirror h_dshape3D; ///< Shape function
                                                 ///< derivatives
};
} // namespace specfem::assembly::mesh_impl
