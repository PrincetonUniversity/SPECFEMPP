#pragma once

namespace specfem {
namespace coordinate_systems {

/**
 * @brief Transform coordinates from one system to another.
 *
 * The return type is the first template parameter (the target coordinate type);
 * the source type is deduced from the argument. Projections that need a
 * configuration (UTM zone, planet radius) provide it as an ordinary overloaded
 * parameter rather than a template parameter -- those overloads are declared in
 * the projection-specific headers (e.g., utm_projection.hpp,
 * geocentric_projection.hpp) alongside their explicit specializations.
 * Unimplemented source/target combinations produce a linker error.
 *
 * @tparam Target Target coordinate type (e.g., cartesian_coordinates)
 * @tparam Source Source coordinate type (deduced)
 * @param source Input coordinates
 * @return Transformed coordinates of type Target
 */
template <typename Target, typename Source>
Target transform(const Source &source);

} // namespace coordinate_systems
} // namespace specfem
