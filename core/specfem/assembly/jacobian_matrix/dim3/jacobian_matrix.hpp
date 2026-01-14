#pragma once

#include "enumerations/interface.hpp"
#include "specfem/data_access.hpp"
#include "specfem/mesh.hpp"
#include <Kokkos_Core.hpp>

namespace specfem::assembly {

// clang-format off
/**
 * @brief 3D Jacobian matrix container for spectral element coordinate transformations
 *
 * The Jacobian matrix enables mapping between physical coordinates (x,y,z) and reference
 * coordinates (ξ,η,γ) for each quadrature point in spectral elements.
 *
 * **Mathematical Foundation:**
 * The Jacobian matrix represents the coordinate transformation:
 * \f[
 * J = \begin{pmatrix}
 * \frac{\partial x}{\partial \xi} & \frac{\partial y}{\partial \xi} & \frac{\partial z}{\partial \xi} \\
 * \frac{\partial x}{\partial \eta} & \frac{\partial y}{\partial \eta} & \frac{\partial z}{\partial \eta} \\
 * \frac{\partial x}{\partial \gamma} & \frac{\partial y}{\partial \gamma} & \frac{\partial z}{\partial \gamma}
 * \end{pmatrix}
 * \f]
 *
 * **Storage Components:**
 * - `xix`, `xiy`, `xiz`: \f$\frac{\partial \xi}{\partial x}\f$, \f$\frac{\partial \xi}{\partial y}\f$, \f$\frac{\partial \xi}{\partial z}\f$ (inverse transformation derivatives)
 * - `etax`, `etay`, `etaz`: \f$\frac{\partial \eta}{\partial x}\f$, \f$\frac{\partial \eta}{\partial y}\f$, \f$\frac{\partial \eta}{\partial z}\f$
 * - `gammax`, `gammay`, `gammaz`: \f$\frac{\partial \gamma}{\partial x}\f$, \f$\frac{\partial \gamma}{\partial y}\f$, \f$\frac{\partial \gamma}{\partial z}\f$
 * - `jacobian`: Determinant \f$|J|\f$ for integration weights
 *
 * @code
 * // Example usage
 * specfem::assembly::jacobian_matrix<specfem::dimension::type::dim3> jacobian(...);
 *
 * // Access in device kernel
 * specfem::point::index<specfem::dimension::type::dim3> idx(ispec, iz, iy, ix);
 * specfem::point::jacobian_matrix<specfem::dimension::type::dim3> point_jac;
 * specfem::assembly::load_on_device(idx, jacobian, point_jac);
 * @endcode
 */
// clang-format on
template <>
struct jacobian_matrix<specfem::dimension::type::dim3>
    : public specfem::data_access::Container<
          specfem::data_access::ContainerType::domain,
          specfem::data_access::DataClassType::jacobian_matrix,
          specfem::dimension::type::dim3> {
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
      specfem::dimension::type::dim3>;

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
   * @brief Number of Gauss-Lobatto-Legendre quadrature points in x-direction
   */
  int ngllx;

  /**
   * @brief Number of Gauss-Lobatto-Legendre quadrature points in y-direction
   */
  int nglly;

  /**
   * @brief Number of Gauss-Lobatto-Legendre quadrature points in z-direction
   */
  int ngllz;

  view_type xix; // ∂ξ/∂x derivatives (device) [nspec][ngllz][nglly][ngllx]
  view_type::HostMirror h_xix; // ∂ξ/∂x derivatives (host)
                               // [nspec][ngllz][nglly][ngllx]
  view_type xiy; // ∂ξ/∂y derivatives (device) [nspec][ngllz][nglly][ngllx]
  view_type::HostMirror h_xiy; // ∂ξ/∂y derivatives (host)
                               // [nspec][ngllz][nglly][ngllx]
  view_type xiz; // ∂ξ/∂z derivatives (device) [nspec][ngllz][nglly][ngllx]
  view_type::HostMirror h_xiz; // ∂ξ/∂z derivatives (host)
                               // [nspec][ngllz][nglly][ngllx]
  view_type etax; // ∂η/∂x derivatives (device) [nspec][ngllz][nglly][ngllx]
  view_type::HostMirror h_etax; // ∂η/∂x derivatives (host)
                                // [nspec][ngllz][nglly][ngllx]
  view_type etay; // ∂η/∂y derivatives (device) [nspec][ngllz][nglly][ngllx]
  view_type::HostMirror h_etay; // ∂η/∂y derivatives (host)
                                // [nspec][ngllz][nglly][ngllx]
  view_type etaz; // ∂η/∂z derivatives (device) [nspec][ngllz][ngllly][ngllx]
  view_type::HostMirror h_etaz; // ∂η/∂z derivatives (host)
                                // [nspec][ngllz][nglly][ngllx]
  view_type gammax; // ∂γ/∂x derivatives (device) [nspec][ngllz][ngllly][ngllx]
  view_type::HostMirror h_gammax; // ∂γ/∂x derivatives (host)
                                  // [nspec][ngllz][nglly][ngllx]
  view_type gammay; // ∂γ/∂y derivatives (device) [nspec][ngllz][ngllly][ngllx]
  view_type::HostMirror h_gammay; // ∂γ/∂y derivatives (host)
                                  // [nspec][ngllz][nglly][ngllx]
  view_type gammaz; // ∂γ/∂z derivatives (device) [nspec][ngllz][ngllly][ngllx]
  view_type::HostMirror h_gammaz;   // ∂γ/∂z derivatives (host)
                                    // [nspec][ngllz][nglly][ngllx]
  view_type jacobian;               // Jacobian determinant (device)
                                    // [nspec][ngllz][nglly][ngllx]
  view_type::HostMirror h_jacobian; // Jacobian determinant (host)
                                    // [nspec][ngllz][nglly][ngllx]

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
   * @param ngllx Number of GLL quadrature points in x-direction
   * @param nglly Number of GLL quadrature points in y-direction
   * @param ngllz Number of GLL quadrature points in z-direction
   */
  jacobian_matrix(const int nspec, const int ngllx, const int nglly,
                  const int ngllz);

  /**
   * @brief Construct Jacobian matrix from assembly mesh
   *
   *
   * @param assembly_mesh Reference to the assembly mesh containing geometric
   *                      and discretization information for the computational
   *                      domain. Must be properly initialized with valid
   *                      spectral element and quadrature point data.
   */
  jacobian_matrix(const specfem::assembly::mesh<dimension_tag> &assembly_mesh);
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
