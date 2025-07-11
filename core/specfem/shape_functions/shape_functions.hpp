#pragma once

#include <vector>

namespace specfem::shape_function {

/** @brief Calculate shape functions for a 2D element given natural
 * coordinates \f$ (\xi, \gamma) \f$
 *
 * @tparam T type of the shape function values (float, double)
 * @param xi \f$ \xi \f$ value of the point
 * @param gamma \f$ \gamma \f$ value of point
 * @param ngod Total number of control nodes per element
 * @return std::vector<T> shape function values at the given point
 */
template <typename T>
std::vector<T> shape_function(const T xi, const T gamma, const int ngod);

/** @brief Calculate shape functions for a 2D element given natural
 * coordinates \f$ (\xi, \gamma) \f$
 *
 * @tparam T type of the shape function values (float, double)
 * @param xi \f$ \xi \f$ value of the point
 * @param gamma \f$ \gamma \f$ value of point
 * @param ngod Total number of control nodes per element
 * @return std::vector<std::vector<T>> shape function values at the given point
 */
template <typename T>
std::vector<std::vector<T> >
shape_function_derivatives(const T xi, const T gamma, const int ngod);

} // namespace specfem::shape_function
