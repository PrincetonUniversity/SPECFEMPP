#pragma once

#include "enumerations/interface.hpp"
#include <Kokkos_Core.hpp>

namespace specfem::assembly::mesh_impl {
/**
 * @brief Shape function and their derivatives for every control node within the
 * mesh
 *
 */
template <> struct shape_functions<specfem::dimension::type::dim2> {

public:
  constexpr static auto dimension_tag =
      specfem::dimension::type::dim2; ///< Dimension
  int ngllz; ///< Number of quadrature points in z dimension
  int ngllx; ///< Number of quadrature points in x dimension
  int ngnod; ///< Number of control nodes

  using ShapeFunctionViewType = Kokkos::View<type_real ***, Kokkos::LayoutRight,
                                             Kokkos::DefaultExecutionSpace>;

  using DShapeFunctionViewType =
      Kokkos::View<type_real ****, Kokkos::LayoutRight,
                   Kokkos::DefaultExecutionSpace>;

  shape_functions(const int &ngllz, const int &ngllx, const int &ngnod)
      : ngllz(ngllz), ngllx(ngllx), ngnod(ngnod),
        shape2D("specfem::assembly::shape_functions::shape2D", ngllz, ngllx,
                ngnod),
        dshape2D("specfem::assembly::shape_functions::dshape2D", ngllz, ngllx,
                 ndim, ngnod),
        h_shape2D(Kokkos::create_mirror_view(shape2D)),
        h_dshape2D(Kokkos::create_mirror_view(dshape2D)) {}

  shape_functions(
      const Kokkos::View<type_real *, Kokkos::DefaultHostExecutionSpace> xi,
      const Kokkos::View<type_real *, Kokkos::DefaultHostExecutionSpace> gamma,
      const int &ngll, const int &ngnod);

  shape_functions() = default;

  ShapeFunctionViewType shape2D;                 ///< Shape functions
  DShapeFunctionViewType dshape2D;               ///< Shape function
                                                 ///< derivatives
  ShapeFunctionViewType::HostMirror h_shape2D;   ///< Shape functions
  DShapeFunctionViewType::HostMirror h_dshape2D; ///< Shape function
                                                 ///< derivatives
};
} // namespace specfem::assembly::mesh_impl
