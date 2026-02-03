#pragma once

#include "domain_view.hpp"
#include "enumerations/interface.hpp"
#include "kokkos_abstractions.h"
#include "specfem/assembly/mesh.hpp"
#include "specfem/data_access.hpp"
#include "specfem/point.hpp"
#include "specfem/quadrature.hpp"
#include "specfem_setup.hpp"
#include <Kokkos_Core.hpp>

namespace specfem::assembly {

// clang-format off
/**
 * @brief 2D Jacobian matrix container for spectral element coordinate transformations
 *
 * The Jacobian matrix enables mapping between physical coordinates (x,z) and reference
 * coordinates (ξ,γ) for each quadrature point in spectral elements.
 *
 * **Mathematical Foundation:**
 * The Jacobian matrix represents the coordinate transformation:
 * \f[
 * J = \begin{pmatrix}
 * \frac{\partial x}{\partial \xi} & \frac{\partial z}{\partial \xi} \\
 * \frac{\partial x}{\partial \gamma} & \frac{\partial z}{\partial \gamma}
 * \end{pmatrix}
 * \f]
 *
 * **Storage Components:**
 * - `xix`: \f$\frac{\partial \xi}{\partial x}\f$ (inverse transformation derivatives)
 * - `xiz`: \f$\frac{\partial \xi}{\partial z}\f$
 * - `gammax`: \f$\frac{\partial \gamma}{\partial x}\f$
 * - `gammaz`: \f$\frac{\partial \gamma}{\partial z}\f$
 * - `jacobian`: Determinant \f$|J|\f$ for integration weights
 *
 * @code
 * // Example usage
 * specfem::assembly::jacobian_matrix<specfem::dimension::type::dim2> jacobian(...);
 *
 * // Access in device kernel
 * specfem::point::index<specfem::dimension::type::dim2> idx(ispec, iz, ix);
 * specfem::point::jacobian_matrix<specfem::dimension::type::dim2> point_jac;
 * specfem::assembly::load_on_device(idx, jacobian, point_jac);
 * @endcode
 */
// clang-format on
template <>
struct jacobian_matrix<specfem::dimension::type::dim2>
    : public specfem::data_access::Container<
          specfem::data_access::ContainerType::domain,
          specfem::data_access::DataClassType::jacobian_matrix,
          specfem::dimension::type::dim2> {
  /**
   * @name Type Definitions
   *
   */
  ///@{

  /**
   * @brief Base container type providing data access infrastructure
   *
   * @see specfem::data_access::Container
   */
  using base_type = specfem::data_access::Container<
      specfem::data_access::ContainerType::domain,
      specfem::data_access::DataClassType::jacobian_matrix,
      specfem::dimension::type::dim2>;

  /**
   * @brief Kokkos view type for storing Jacobian matrix components
   */
  using view_type = typename base_type::scalar_type<
      type_real, Kokkos::DefaultExecutionSpace::memory_space>;
  ///@}

  /**
   * @brief Number of spectral elements in the computational domain
   */
  int nspec;

  /**
   * @brief Number of Gauss-Lobatto-Legendre quadrature points in z-direction
   */
  int ngllz;

  /**
   * @brief Number of Gauss-Lobatto-Legendre quadrature points in x-direction
   */
  int ngllx;

  view_type xix; // ∂ξ/∂x derivatives (device) [nspec][ngllz][ngllx]
  view_type::HostMirror h_xix; // ∂ξ/∂x derivatives (host) [nspec][ngllz][ngllx]
  view_type xiz; // ∂ξ/∂z derivatives (device) [nspec][ngllz][ngllx]
  view_type::HostMirror h_xiz; // ∂ξ/∂z derivatives (host) [nspec][ngllz][ngllx]
  view_type gammax; // ∂γ/∂x derivatives (device) [nspec][ngllz][ngllx]
  view_type::HostMirror h_gammax; // ∂γ/∂x derivatives (host)
                                  // [nspec][ngllz][ngllx]
  view_type gammaz; // ∂γ/∂z derivatives (device) [nspec][ngllz][ngllx]
  view_type::HostMirror h_gammaz; // ∂γ/∂z derivatives (host)
                                  // [nspec][ngllz][ngllx]
  view_type jacobian; // Jacobian determinant (device) [nspec][ngllz][ngllx]
  view_type::HostMirror h_jacobian; // Jacobian determinant (host)
                                    // [nspec][ngllz][ngllx]

  /**
   * @name Constructors
   *
   * Object lifecycle management for Jacobian matrix data structures.
   */
  ///@{

  /**
   * @brief Default constructor
   *
   * Creates an empty Jacobian matrix container. Use parameterized constructors
   * to initialize with actual mesh data and compute transformation derivatives.
   */
  jacobian_matrix() = default;

  /**
   * @brief Construct Jacobian matrix with specified dimensions
   *
   * Allocates storage for Jacobian matrix components based on problem
   * dimensions and quadrature point specifications. The actual derivative
   * values must be computed and populated separately.
   *
   * @param nspec Number of spectral elements in the domain
   * @param ngllz Number of GLL quadrature points in z-direction
   * @param ngllx Number of GLL quadrature points in x-direction
   */
  jacobian_matrix(const int nspec, const int ngllz, const int ngllx);

  /**
   * @brief Construct Jacobian matrix from 2D mesh information
   *
   * Computes and initializes all Jacobian matrix components from mesh geometry.
   * This constructor:
   * - Computes coordinate transformation derivatives at all quadrature points
   * - Calculates Jacobian determinants for integration weights
   * - Sets up efficient device/host memory layouts
   * - Validates mesh geometry and transformation quality
   *
   * @param mesh 2D finite element mesh containing element connectivity,
   *             node coordinates, and geometric information
   *
   * @code
   * specfem::assembly::mesh<specfem::dimension::type::dim2> mesh;
   * // ... initialize mesh
   * specfem::assembly::jacobian_matrix<specfem::dimension::type::dim2>
   * jac(mesh);
   * @endcode
   */
  jacobian_matrix(
      const specfem::assembly::mesh<specfem::dimension::type::dim2> &mesh);
  ///@}

  /**
   * @brief Synchronize device and host memory views
   *
   * Ensures consistency between device and host copies of Jacobian matrix
   * data by copying from device to host. This is typically called after
   * device-based computations to make results available on the host.
   */
  void sync_views();

  /**
   * @brief Validate Jacobian determinant values for numerical stability
   *
   * Checks all Jacobian determinant values to identify elements with
   * potentially problematic transformations that could cause numerical
   * instability during computation. Small or negative Jacobians indicate
   * degenerate or inverted elements.
   *
   * @return std::tuple containing:
   *         - Boolean flag indicating if any small Jacobians were found
   *         - Boolean array flagging elements with small Jacobian values
   *
   * @note Small Jacobians are typically defined as values below a threshold
   *       that could cause numerical instability in the finite element
   * computation.
   *
   * @code
   * auto [has_small_jac, element_flags] = jacobian.check_small_jacobian();
   * if (has_small_jac) {
   *     // Handle problematic elements
   *     for (int i = 0; i < nspec; ++i) {
   *         if (element_flags(i)) {
   *             std::cout << "Element " << i << " has small Jacobian" <<
   * std::endl;
   *         }
   *     }
   * }
   * @endcode
   */
  std::tuple<bool, Kokkos::View<bool *, Kokkos::DefaultHostExecutionSpace> >
  check_small_jacobian() const;
};
} // namespace specfem::assembly
