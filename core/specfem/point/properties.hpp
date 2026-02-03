#pragma once

#include "enumerations/interface.hpp"
#include "specfem/setup.hpp"

#include "medium/dim2/acoustic/isotropic/properties.hpp"
#include "medium/dim2/elastic/anisotropic/properties.hpp"
#include "medium/dim2/elastic/isotropic/properties.hpp"
#include "medium/dim2/elastic/isotropic_cosserat/properties.hpp"
#include "medium/dim2/electromagnetic/isotropic/properties.hpp"
#include "medium/dim2/poroelastic/isotropic/properties.hpp"

namespace specfem {
namespace point {

/**
 * @brief Properties of a quadrature point.
 *
 * This class serves as a container for physical properties at a specific quadrature point
 * within an element. It is templated on dimension, medium type, and property type to
 * provide specialized storage and accessors for different physical models (e.g.,
 * acoustic, elastic, poroelastic).
 *
 * @tparam Dimension The dimension of the medium (e.g., dim2, dim3)
 * @tparam MediumTag The type of the medium (e.g., acoustic, elastic)
 * @tparam PropertyTag The type of the properties (e.g., isotropic, anisotropic)
 * @tparam UseSIMD Boolean indicating whether to use SIMD intrinsics for storage and operations
 *
 * @note Medium-specific specializations are available in the implementation
 * details. See @ref specfem::point::impl::properties::data_container.
 *
 * @section usage Usage Example
 *
 * The following example demonstrates how to instantiate and use the properties class
 * for a 2D elastic isotropic medium:
 *
 * @code
 * #include "specfem/point/properties.hpp"
 *
 * // Define types
 * using namespace specfem::element;
 * constexpr auto dim = specfem::dimension::type::dim2;
 * constexpr bool use_simd = false;
 *
 * // Instantiate properties for 2D Elastic Isotropic medium
 * // Requires Bulk Modulus (kappa), Shear Modulus (mu), and Density (rho)
 * double kappa = 20.0;
 * double mu = 10.0;
 * double rho = 2500.0;
 *
 * specfem::point::properties<dim, medium_tag::elastic, property_tag::isotropic, use_simd>
 *     props(kappa, mu, rho);
 *
 * // Access properties
 * double k = props.kappa();
 * double m = props.mu();
 * double r = props.rho();
 *
 * // Derived properties are also available
 * double l2m = props.lambdaplus2mu(); // lambda + 2*mu
 * @endcode
 *
 * @section see_also See Also
 * - specfem::point::impl::properties::data_container
 * - specfem::compute::mass_matrix
 */
template <specfem::dimension::type Dimension,
          specfem::element::medium_tag MediumTag,
          specfem::element::property_tag PropertyTag, bool UseSIMD>
struct properties : impl::properties::data_container<Dimension, MediumTag,
                                                     PropertyTag, UseSIMD> {

  using base_type = impl::properties::data_container<Dimension, MediumTag,
                                                     PropertyTag, UseSIMD>;

  using value_type = typename base_type::value_type;
  using simd = typename base_type::simd;

  using base_type::base_type;
};

} // namespace point
} // namespace specfem
