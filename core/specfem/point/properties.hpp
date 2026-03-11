#pragma once

#include "specfem/enums.hpp"
#include "specfem/medium_container/point_properties.hpp"
#include "specfem/setup.hpp"
#include "specfem/tags.hpp"

namespace specfem::point {
namespace impl {

/// @private Implementation detail
template <specfem::element::dimension_tag DimensionTag,
          specfem::element::medium_tag MediumTag,
          specfem::element::property_tag PropertyTag, bool UseSIMD>
struct properties : specfem::medium_container::properties::point_container<
                        DimensionTag, MediumTag, PropertyTag, UseSIMD> {

  using base_type = specfem::medium_container::properties::point_container<
      DimensionTag, MediumTag, PropertyTag, UseSIMD>;

  using value_type = typename base_type::value_type;
  using simd = typename base_type::simd;

  using base_type::base_type;
};

} // namespace impl

/**
 * @brief Properties of a quadrature point.
 *
 * This class serves as a container for physical properties at a specific
 * quadrature point within an element. It is templated on dimension, medium
 * type, and property type to provide specialized storage and accessors for
 * different physical models (e.g., acoustic, elastic, poroelastic).
 *
 * @tparam Tags The tags for the element where the quadrature point is located
 *
 * @note Medium-specific specializations are available in the implementation
 * details. See @ref specfem::medium_container::properties::point_container.
 *
 * @section usage Usage Example
 *
 * The following example demonstrates how to instantiate and use the properties
 * class for a 2D elastic isotropic medium:
 *
 * @code
 * #include "specfem/point/properties.hpp"
 *
 * // Define types
 * using namespace specfem::element;
 * constexpr auto dim = specfem::element::dimension_tag::dim2;
 * constexpr bool use_simd = false;
 *
 * // Instantiate properties for 2D Elastic Isotropic medium
 * // Requires Bulk Modulus (kappa), Shear Modulus (mu), and Density (rho)
 * double kappa = 20.0;
 * double mu = 10.0;
 * double rho = 2500.0;
 *
 * specfem::point::properties<specfem::tags::Tags<
 *     dim, medium_tag::elastic, property_tag::isotropic, use_simd>>
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
 * - specfem::medium_container::properties::point_container
 * - specfem::compute::mass_matrix
 */
template <typename Tags>
using properties = impl::properties<Tags::dimension_tag, Tags::medium_tag,
                                    Tags::property_tag, Tags::using_simd>;
} // namespace specfem::point
