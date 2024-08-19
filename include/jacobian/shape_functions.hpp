#ifndef _SHAPE_FUNCTIONS_HPP
#define _SHAPE_FUNCTIONS_HPP

#include "kokkos_abstractions.h"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace jacobian {

/**
 * @brief Compute shape functions at particular point (xi, gamma)
 *
 * @note Use this function only outside of parallel region
 *
 * @param xi \f$ \xi \f$ value of the point
 * @param gamma \f$ \gamma \f$ value of the point
 * @param ngod Total number of control nodes per element
 * @return specfem::kokkos::HostView1d<type_real> View defining the shape
 * function
 */
specfem::kokkos::HostView1d<type_real>
define_shape_functions(const type_real xi, const type_real gamma,
                       const int ngod);
/**
 * @brief Derivates of shape function at a particular point
 *
 * @note Use this function only outside of parallel region
 *
 * @param xi \f$ \xi \f$ value of the point
 * @param gamma \f$ \gamma \f$ value of the point
 * @param ngod Total number of control nodes per element
 * @return specfem::kokkos::HostView2d<type_real> View defining the derivative
 * of shape function (\f$ \partial N/\partial \xi \f$, \f$ \partial N/\partial
 * \gamma \f$)
 */
specfem::kokkos::HostView2d<type_real>
define_shape_functions_derivatives(const type_real xi, const type_real gamma,
                                   const int ngod);

/**
 * @brief Compute shape functions at particular point (xi, gamma)
 *
 * @param xi \f$ \xi \f$ value of the point
 * @param gamma \f$ \gamma \f$ value of the point
 * @param ngod Total number of control nodes per element
 * @param shape2D View defining the shape function (updated by this function)
 */
void define_shape_functions(specfem::kokkos::HostView1d<type_real> shape2D,
                            const type_real xi, const type_real gamma,
                            const int ngod);
/**
 * @brief Derivates of shape function at a particular point
 *
 * @param xi \f$ \xi \f$ value of the point
 * @param gamma \f$ \gamma \f$ value of the point
 * @param ngod Total number of control nodes per element
 * @param dershape2D View defining the derivatives of shape
 * function (\f$ \partial N/\partial \xi \f$, \f$ \partial N/\partial \gamma
 * \f$) (updated by this function)
 */
void define_shape_functions_derivatives(
    specfem::kokkos::HostView2d<type_real> dershape2D, const type_real xi,
    const type_real gamma, const int ngod);
} // namespace jacobian
} // namespace specfem

#endif // SHAPE_FUNCTIONS_H
