#pragma once

#include <type_traits>
#include <vector>

namespace specfem::shape_function {

/** @brief Calculate shape functions for a 3D element given natural
 * coordinates \f$ (\xi, \eta, \zeta) \f$
 *
 * @tparam T type of the shape function values (float, double)
 * @param xi \f$ \xi \f$ value of the point
 * @param eta \f$ \eta \f$ value of point
 * @param zeta \f$ \zeta \f$ value of point
 * @param ngod Total number of control nodes per element
 * @return std::vector<T> shape function values at the given point
 */
template <typename T>
std::vector<T> shape_function(const T xi, const T eta, const T zeta,
                              const int ngod);

/** @brief Calculate shape function derivatives for a 3D element
 * given natural coordinates \f$ (\xi, \eta, \zeta) \f
 *
 * @tparam T type of the shape function values (float, double)
 * @param xi \f$ \xi \f$ value of the point
 * @param eta \f$ \eta \f$ value of point
 * @param zeta \f$ \zeta \f$ value of point
 * @param ngod Total number of control nodes per element
 * @return std::vector<std::vector<T>> shape function derivatives at the given
 * point
 */
template <typename T>
std::vector<std::vector<T> > shape_function_derivatives(const T xi, const T eta,
                                                        const T zeta,
                                                        const int ngod);

} // namespace specfem::shape_function
