#pragma once

#include "impl/field.hpp"
#include "specfem/data_access.hpp"

namespace specfem::point {

/**
 * @brief Displacement field accessor
 *
 * This class provides a type-safe, high-performance interface for accessing and
 * manipulating displacement field data at individual quadrature points within
 * spectral elements. Displacement \f$\mathbf{u}\f$ is the fundamental kinematic
 * quantity in solid mechanics and elastic wave propagation.
 *
 * @tparam DimensionTag The spatial dimension defining displacement vector size:
 *                      - `specfem::dimension::type::dim2`: 2D problems (ux, uz)
 *                      - `specfem::dimension::type::dim3`: 3D problems (ux, uy,
 * uz)
 *
 * @tparam MediumTag The physical medium type governing displacement behavior:
 *                   - `specfem::element::medium_tag::acoustic`: Often
 * represents pressure potential
 *                   - `specfem::element::medium_tag::elastic`: True mechanical
 * displacement
 *                   - `specfem::element::medium_tag::poroelastic`: Solid phase
 * displacement in Biot theory
 *
 * @tparam UseSIMD Boolean flag controlling SIMD vectorization for processing
 * multiple quadrature points simultaneously. Essential for performance in
 *                 large-scale simulations with millions of degrees of freedom.
 *
 * @note Displacement fields are often subject to boundary conditions (Dirichlet
 * for prescribed displacement, Neumann for prescribed traction) which must be
 *       enforced during the solution process.
 *
 * @see specfem::point::velocity
 * @see specfem::point::acceleration
 * @see specfem::point::stress
 * @see specfem::boundary_conditions
 *
 * @code
 * // Example: Creating 3D elastic displacement field for structural analysis
 * using DispField = specfem::point::displacement<
 *     specfem::dimension::type::dim3,
 *     specfem::element::medium_tag::elastic,
 *     false>;  // Scalar operations for precise debugging
 *
 * DispField disp(0.0);  // Initialize at reference configuration
 *
 * // Set displacement components (e.g., from boundary conditions)
 * disp(0) = 0.001;  // 1 mm displacement in x-direction
 * disp(1) = 0.002;  // 2 mm displacement in y-direction
 * disp(2) = 0.003;  // 3 mm displacement in z-direction
 *
 * // Compute displacement magnitude for analysis
 * auto magnitude = std::sqrt(disp(0)*disp(0) + disp(1)*disp(1) +
 * disp(2)*disp(2));
 * @endcode
 *
 * @code
 * // Example: Time integration using displacement field
 * DispField displacement, displacement_old;
 * VelocityField velocity;
 * AccelField acceleration;
 *
 * // Newmark time integration scheme
 * const auto dt = 0.001;  // Time step
 * const auto beta = 0.25; // Newmark parameter
 *
 * // Predictor step
 * for (int comp = 0; comp < displacement.components; ++comp) {
 *   displacement(comp) = displacement_old(comp) + velocity(comp) * dt
 *                      + 0.5 * (1 - 2*beta) * acceleration(comp) * dt * dt;
 * }
 * @endcode
 *
 * @code
 * // Example: Strain computation from displacement gradients
 * DispField displacement;
 * specfem::assembly::load_on_device(index, fields, displacement);
 *
 * // Compute displacement gradients (requires element geometry)
 * auto grad_u = compute_displacement_gradients(displacement, jacobian_matrix);
 *
 * // Strain tensor components (2D example)
 * auto eps_xx = grad_u[0][0];                           // ∂u_x/∂x
 * auto eps_zz = grad_u[1][1];                           // ∂u_z/∂z
 * auto eps_xz = 0.5 * (grad_u[0][1] + grad_u[1][0]);   // 0.5(∂u_x/∂z +
 * ∂u_z/∂x)
 * @endcode
 *
 * @code
 * // Example: Boundary condition enforcement
 * DispField displacement;
 *
 * // Apply Dirichlet boundary condition (prescribed displacement)
 * if (is_dirichlet_boundary(point_index)) {
 *   auto prescribed_value = get_boundary_displacement(point_index, time);
 *   displacement(component) = prescribed_value;
 * }
 *
 * // Store displacement for output (seismograms, visualization)
 * specfem::io::write_displacement_field(displacement, output_time);
 * @endcode
 *
 *
 * // Access displacement components for strain computation
 * auto ux = displacement(0);
 * auto uy = displacement(1);
 * auto uz = displacement(2);
 *
 * // Strain components would be computed from spatial derivatives
 * // (implementation depends on quadrature and differentiation operators)
 * @endcode'
 *
 * @see specfem::point::velocity for velocity field accessor
 * @see specfem::point::acceleration for acceleration field accessor
 */
template <specfem::dimension::type DimensionTag,
          specfem::element::medium_tag MediumTag, bool UseSIMD>
class displacement
    : public impl::field<DimensionTag, MediumTag,
                         specfem::data_access::DataClassType::displacement,
                         UseSIMD> {
private:
  /// @brief Type alias for the base field implementation
  using base_type =
      impl::field<DimensionTag, MediumTag,
                  specfem::data_access::DataClassType::displacement, UseSIMD>;

public:
  /// @brief SIMD type for vectorized displacement operations
  using simd = typename base_type::simd;

  /// @brief Vector type for storing displacement component values
  using value_type = typename base_type::value_type;

  /// @brief Inherit all constructors from the base field implementation
  using base_type::base_type;
};

} // namespace specfem::point
