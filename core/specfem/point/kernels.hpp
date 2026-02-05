#pragma once

#include "specfem/enums.hpp"
#include "specfem/medium_container.hpp"

namespace specfem::point {

/**
 * @brief Kernels of a quadrature point.
 *
 * This class serves as a container for sensitivity (misfit) kernels at a
 * specific quadrature point within an element. These kernels represent the
 * gradient of the misfit function with respect to physical parameters (e.g.,
 * density, velocity, moduli) and are essential for adjoint tomography and
 * inversion workflows.
 *
 * It is templated on dimension, medium type, and property type to provide
 * specialized storage and accessors for different physical models.
 *
 * @tparam DimensionTag The dimension of the medium (e.g., dim2, dim3)
 * @tparam MediumTag The type of the medium (e.g., acoustic, elastic)
 * @tparam PropertyTag The type of the kernels (e.g., isotropic, anisotropic)
 * @tparam UseSIMD Boolean indicating whether to use SIMD intrinsics
 *
 * @note Medium-specific specializations are available in the implementation
 * details. See @ref specfem::medium_container::kernels::point_container.
 *
 * @section usage Usage Example
 *
 * The following example demonstrates how to instantiate and use the kernels
 * class for a 2D elastic isotropic medium. In this context, the stored values
 * represent the sensitivity densities (e.g., K_rho, K_mu, K_kappa).
 *
 * @code
 * #include "specfem/point/kernels.hpp"
 *
 * // Define types
 * using namespace specfem::element;
 * constexpr auto dim = specfem::element::dimension_tag::dim2;
 * constexpr bool use_simd = false;
 *
 * // Instantiate kernels for 2D Elastic Isotropic medium
 * // Values represent sensitivity kernels for:
 * // rho (density), mu (shear modulus), kappa (bulk modulus),
 * // rhop (density perturbation?), alpha (P-velocity), beta (S-velocity)
 * double K_rho = 0.1;
 * double K_mu = 0.2;
 * double K_kappa = 0.3;
 * double K_rhop = 0.0;
 * double K_alpha = 0.5;
 * double K_beta = 0.4;
 *
 * specfem::point::kernels<dim, medium_tag::elastic, property_tag::isotropic,
 * use_simd> kernels(K_rho, K_mu, K_kappa, K_rhop, K_alpha, K_beta);
 *
 * // Access kernel values
 * double k_mu = kernels.mu();
 * double k_alpha = kernels.alpha();
 * @endcode
 *
 * @section see_also See Also
 * - specfem::medium_container::kernels::point_container
 * - specfem::compute::frechet_derivative
 */
template <specfem::element::dimension_tag DimensionTag,
          specfem::element::medium_tag MediumTag,
          specfem::element::property_tag PropertyTag, bool UseSIMD,
          typename Enable = void>
struct kernels : public specfem::medium_container::kernels::point_container<
                     DimensionTag, MediumTag, PropertyTag, UseSIMD> {
  using base_type = specfem::medium_container::kernels::point_container<
      DimensionTag, MediumTag, PropertyTag, UseSIMD>;

  using value_type = typename base_type::value_type;
  using simd = typename base_type::simd;

  using base_type::base_type;
  constexpr static auto dimension_tag =
      DimensionTag;                                 ///< dimension of the medium
  constexpr static auto medium_tag = MediumTag;     ///< type of the medium
  constexpr static auto property_tag = PropertyTag; ///< type of the properties
};

} // namespace specfem::point
