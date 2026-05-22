#pragma once

namespace specfem {
namespace coordinate_systems {

/**
 * @brief Transform coordinates from one system to another.
 *
 * The return type is the first template parameter (the target coordinate type).
 * Source and config types are deduced from the arguments. Unimplemented
 * source/target/config combinations produce a linker error.
 *
 * Projection-specific headers (e.g., utm.hpp) declare explicit specializations.
 *
 * @tparam Target Target coordinate type (e.g., cartesian_coordinates)
 * @tparam Source Source coordinate type (deduced)
 * @tparam Config Projection configuration type (deduced)
 * @param source Input coordinates
 * @param config Projection configuration
 * @return Transformed coordinates of type Target
 */
template <typename Target, typename Source, typename Config>
Target transform(const Source &source, const Config &config);

/**
 * @brief Config-free overload for purely mathematical transforms.
 *
 * @tparam Target Target coordinate type
 * @tparam Source Source coordinate type (deduced)
 * @param source Input coordinates
 * @return Transformed coordinates of type Target
 */
template <typename Target, typename Source>
Target transform(const Source &source);

} // namespace coordinate_systems
} // namespace specfem
