#ifndef SHAPE_FUNCTIONS_H
#define SHAPE_FUNCTIONS_H

#include "../include/kokkos_abstractions.h"
#include <Kokkos_Core.hpp>

namespace shape_functions {

/**
 * @brief Compute shape functions at particular point (xi, gamma)
 *
 * @note Use this function only outside of parallel region
 *
 * @param xi \f$ \xi \f$ value of the point
 * @param gamma \f$ \gamma \f$ value of the point
 * @param ngod Total number of control nodes per element
 * @return specfem::HostView1d<type_real> View defining the shape function
 */
specfem::HostView1d<type_real>
define_shape_functions(const double xi, const double gamma, const int ngod);
/**
 * @brief Derivates of shape function at a particular point
 *
 * @note Use this function only outside of parallel region
 *
 * @param xi \f$ \xi \f$ value of the point
 * @param gamma \f$ \gamma \f$ value of the point
 * @param ngod Total number of control nodes per element
 * @return specfem::HostView2d<type_real> View defining the derivative of shape
 * function (\f$ \partial N/\partial \xi \f$, \f$ \partial N/\partial \gamma
 * \f$)
 */
specfem::HostView2d<type_real>
define_shape_functions_derivatives(const double xi, const double gamma,
                                   const int ngod);

/**
 * @brief Compute shape functions at particular point (xi, gamma)
 *
 * @param xi \f$ \xi \f$ value of the point
 * @param gamma \f$ \gamma \f$ value of the point
 * @param ngod Total number of control nodes per element
 * @param shape2D View defining the shape function (updated by this function)
 */
void define_shape_functions(specfem::HostView1d<type_real> shape2D,
                            const double xi, const double gamma,
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
    specfem::HostView2d<type_real> dershape2D, const double xi,
    const double gamma, const int ngod);
} // namespace shape_functions

#endif // SHAPE_FUNCTIONS_H
