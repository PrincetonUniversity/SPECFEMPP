#pragma once

#include "impl/attenuation_values.hpp"
#include "specfem/enums.hpp"
#include "specfem/point/properties.hpp"
#include "specfem/setup.hpp"
#include <ostream>
#include <tuple>

namespace specfem::medium_container {

/**
 * @brief Template for material properties in seismic simulations.
 *
 * This template stores physical material parameters (density, elastic moduli,
 * wave speeds) and converts them to computational point properties. Different
 * specializations handle various combinations of spatial dimension, medium
 * type, and material properties.
 *
 * Each specialization provides:
 * - Constructor accepting medium-specific physical parameters
 * - get_properties() method returning specfem::point::properties
 * - Equality comparison operators (==, !=)
 *
 * Example usage:
 * @code
 * // Create 2D elastic isotropic material
 * using Mat = specfem::medium_container::material<
 *     specfem::element::dimension_tag::dim2,
 *     specfem::element::medium_tag::elastic,
 *     specfem::element::property_tag::isotropic>;
 *
 * Mat material(lambda, mu, density);
 * auto properties = material.get_properties();
 * @endcode
 *
 * @tparam dimension_tag Spatial dimension (dim2 or dim3)
 * @tparam MediumTag Medium type (acoustic, elastic, etc.)
 * @tparam PropertyTag Property type (isotropic, anisotropic, etc.)
 * @tparam AttenuationTag Attenuation model (no_attenuation, constant_q, etc.)
 * @tparam Enable SFINAE parameter for template specialization
 *
 * @note This stores properties for a domain section. For GLL-level properties,
 * use specfem::assembly::properties.
 */
template <specfem::element::dimension_tag dimension_tag,
          specfem::element::medium_tag MediumTag,
          specfem::element::property_tag PropertyTag,
          specfem::element::attenuation_tag AttenuationTag,
          typename Enable = void>
class material;

} // namespace specfem::medium_container

#include "specfem/medium/dim2/acoustic/isotropic/material.hpp"
#include "specfem/medium/dim2/elastic/anisotropic/material.hpp"
#include "specfem/medium/dim2/elastic/isotropic/material.hpp"
#include "specfem/medium/dim2/elastic/isotropic_cosserat/material.hpp"
#include "specfem/medium/dim2/electromagnetic/isotropic/material.hpp"
#include "specfem/medium/dim2/poroelastic/isotropic/material.hpp"

#include "specfem/medium/dim3/acoustic/isotropic/material.hpp"
#include "specfem/medium/dim3/elastic/isotropic/material.hpp"
