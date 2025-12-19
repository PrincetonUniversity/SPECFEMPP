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
  /**
   * @brief Compile-time dimension tag for template specialization.
   */
  constexpr static auto dimension_tag =
      specfem::dimension::type::dim2; ///< Dimension

  int ngllz; ///< Number of quadrature points in z dimension

  int ngllx; ///< Number of quadrature points in x dimension

  int ngnod; ///< Number of control nodes

  /**
   * @brief Kokkos view type for shape function storage.
   *
   * 3D view with dimensions [ngllz, ngllx, ngnod] using right layout
   * for optimal memory access when iterating over control nodes.
   */
  using ShapeFunctionViewType = Kokkos::View<type_real ***, Kokkos::LayoutRight,
                                             Kokkos::DefaultExecutionSpace>;

  /**
   * @brief Kokkos view type for shape function derivative storage.
   *
   * 4D view with dimensions [ngllz, ngllx, ndim, ngnod] where ndim=2,
   * storing derivatives with respect to both \f$\xi\f$ and \f$\zeta\f$
   * directions.
   */
  using DShapeFunctionViewType =
      Kokkos::View<type_real ****, Kokkos::LayoutRight,
                   Kokkos::DefaultExecutionSpace>;

  /**
   * @brief Constructor for pre-allocated shape function computation.
   *
   * Creates shape function storage with the specified dimensions but does not
   * compute values. This constructor is used when shape functions will be
   * computed and populated separately, typically in mesh assembly operations.
   *
   * @param ngllz Number of GLL quadrature points in z direction
   * @param ngllx Number of GLL quadrature points in x direction
   * @param ngnod Number of control nodes per element
   *
   */
  shape_functions(const int &ngllz, const int &ngllx, const int &ngnod)
      : ngllz(ngllz), ngllx(ngllx), ngnod(ngnod),
        shape2D("specfem::assembly::shape_functions::shape2D", ngllz, ngllx,
                ngnod),
        dshape2D("specfem::assembly::shape_functions::dshape2D", ngllz, ngllx,
                 ndim, ngnod),
        h_shape2D(Kokkos::create_mirror_view(shape2D)),
        h_dshape2D(Kokkos::create_mirror_view(dshape2D)) {}

  /**
   * @brief Constructor with immediate shape function computation.
   *
   * Computes shape functions and their derivatives at the provided quadrature
   * points using the spectral element shape function library. This constructor
   * performs parallel computation using Kokkos and synchronizes data between
   * host and device memory.
   *
   * @param xi GLL quadrature points in the xi (x) direction on [-1,1]
   * @param gamma GLL quadrature points in the gamma (z) direction on [-1,1]
   * @param ngll Number of GLL points (assumed equal in both directions)
   * @param ngnod Number of control nodes per element
   */
  shape_functions(
      const Kokkos::View<type_real *, Kokkos::DefaultHostExecutionSpace> xi,
      const Kokkos::View<type_real *, Kokkos::DefaultHostExecutionSpace> gamma,
      const int &ngll, const int &ngnod);

  /**
   * @brief Default constructor.
   */
  shape_functions() = default;

  /**
   * @brief Device view containing shape function values.
   *
   */
  ShapeFunctionViewType shape2D; ///< Shape functions

  /**
   * @brief Device view containing shape function derivatives.
   *
   */
  DShapeFunctionViewType dshape2D; ///< Shape function
                                   ///< derivatives

  /**
   * @brief Host mirror view of shape function values.
   */
  ShapeFunctionViewType::HostMirror h_shape2D; ///< Shape functions

  /**
   * @brief Host mirror view of shape function derivatives.
   */
  DShapeFunctionViewType::HostMirror h_dshape2D; ///< Shape function
                                                 ///< derivatives

private:
  /**
   * @brief Number of spatial dimensions (always 2 for this specialization).
   */
  constexpr static int ndim = 2;
};
} // namespace specfem::assembly::mesh_impl
