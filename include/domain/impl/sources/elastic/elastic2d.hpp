#ifndef _DOMAIN_SOURCE_ELASTIC2D_HPP
#define _DOMAIN_SOURCE_ELASTIC2D_HPP

#include "domain/impl/sources/source.hpp"
#include "specfem_enums.hpp"
#include "specfem_setup.hpp"
#include <Kokkos_Core.hpp>

using field_type = Kokkos::Subview<
    specfem::kokkos::DeviceView2d<type_real, Kokkos::LayoutLeft>, int,
    std::remove_const_t<decltype(Kokkos::ALL)> >;

namespace specfem {
namespace domain {
namespace impl {
namespace sources {
/**
 * @brief Elemental source class for 2D elastic media.
 *
 * Base class for all 2D elastic elemental sources. This class contains pure
 * virtual methods that must be implemented by the template specializations.
 *
 * @tparam quadrature_points Number of Gauss-Lobatto-Legendre quadrature points
 * defined at compile time or at runtime
 */
template <typename quadrature_points>
class source<specfem::enums::element::dimension::dim2,
             specfem::enums::element::medium::elastic, quadrature_points> {
public:
  /**
   * @brief Compute the source time function value at a given time
   *
   * @param t Time
   * @return type_real Source time function value
   */
  KOKKOS_INLINE_FUNCTION virtual type_real
  eval_stf(const type_real &t) const = 0;

  /**
   * @brief Compute elemental source contribution to the global force vector
   *
   * @param xz Index of the quadrature point
   * @param stf_value Source time function value
   * @param accelx Acceleration in the x direction (return value)
   * @param accelz Acceleration in the z direction (return value)
   */
  KOKKOS_INLINE_FUNCTION virtual void
  compute_interaction(const int &xz, const type_real &stf_value,
                      type_real *accelx, type_real *accelz) const = 0;

  /**
   * @brief Update the acceleration field
   *
   * @param accelx Acceleration in the x direction as computed by
   * compute_interaction
   * @param accelz Acceleration in the z direction as computed by
   * compute_interaction
   * @param field_dot_dot Acceleration field subviewed at global index
   * ibool(ispec, iz, ix)
   */
  KOKKOS_INLINE_FUNCTION virtual void
  update_acceleration(const type_real &accelx, const type_real &accelz,
                      field_type field_dot_dot) const = 0;

  /**
   * @brief Get the ispec index of the source
   *
   * @return int ispec index of the source
   */
  KOKKOS_INLINE_FUNCTION virtual int get_ispec() const = 0;
};

} // namespace sources
} // namespace impl
} // namespace domain
} // namespace specfem

#endif
