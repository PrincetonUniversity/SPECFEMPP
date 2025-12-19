#pragma once

#include "enumerations/interface.hpp"
#include <Kokkos_Core.hpp>

namespace specfem::assembly::mesh_impl {
/**
 * @brief 3D shape functions and derivatives for spectral elements.
 *
 * Computes and stores shape function values and derivatives at all
 * GLL quadrature points for 3D hexahedral elements.
 *
 * @see specfem::shape_function::shape_function
 */
template <> struct shape_functions<specfem::dimension::type::dim3> {
private:
  constexpr static int ndim = 3;

public:
  constexpr static auto dimension_tag =
      specfem::dimension::type::dim3; ///< Dimension tag

  int ngllz; ///< Number of GLL points in z direction
  int nglly; ///< Number of GLL points in y direction
  int ngllx; ///< Number of GLL points in x direction
  int ngnod; ///< Number of control nodes per element

  /**
   * @brief Shape function view type.
   *
   * Dimensions: [ngllz, nglly, ngllx, ngnod].
   */
  using ShapeFunctionViewType =
      Kokkos::View<type_real ****, Kokkos::LayoutRight,
                   Kokkos::DefaultExecutionSpace>;

  /**
   * @brief Shape function derivative view type.
   *
   * Dimensions: [ngllz, nglly, ngllx, ndim, ngnod].
   */
  using DShapeFunctionViewType =
      Kokkos::View<type_real *****, Kokkos::LayoutRight,
                   Kokkos::DefaultExecutionSpace>;

  /**
   * @brief Constructor with pre-allocated storage.
   *
   * Allocates views but does not compute values.
   */
  shape_functions(const int &ngllz, const int &nglly, const int &ngllx,
                  const int &ngnod)
      : ngllz(ngllz), nglly(nglly), ngllx(ngllx), ngnod(ngnod),
        shape3D("specfem::assembly::shape_functions::shape2D", ngllz, nglly,
                ngllx, ngnod),
        dshape3D("specfem::assembly::shape_functions::dshape2D", ngllz, nglly,
                 ngllx, ndim, ngnod),
        h_shape3D(Kokkos::create_mirror_view(shape3D)),
        h_dshape3D(Kokkos::create_mirror_view(dshape3D)) {}

  /**
   * @brief Default constructor.
   */
  shape_functions() = default;

  /**
   * @brief Constructor with immediate computation.
   *
   * Computes shape functions and derivatives from quadrature points.
   */
  shape_functions(const int ngllz, const int nglly, const int ngllx,
                  const int ngnod,
                  const specfem::assembly::mesh_impl::quadrature<
                      specfem::dimension::type::dim3> &quadrature,
                  const specfem::assembly::mesh_impl::control_nodes<
                      specfem::dimension::type::dim3>
                      control_nodes);

  ShapeFunctionViewType shape3D;                 ///< Device shape functions
  DShapeFunctionViewType dshape3D;               ///< Device derivatives
  ShapeFunctionViewType::HostMirror h_shape3D;   ///< Host shape functions
  DShapeFunctionViewType::HostMirror h_dshape3D; ///< Host derivatives
};
} // namespace specfem::assembly::mesh_impl
