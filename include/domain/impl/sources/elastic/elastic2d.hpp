#ifndef _DOMAIN_SOURCE_ELASTIC2D_HPP
#define _DOMAIN_SOURCE_ELASTIC2D_HPP

#include "domain/impl/sources/source.hpp"
#include "specfem_enums.hpp"
#include "specfem_setup.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace domain {
namespace impl {
namespace sources {
template <typename quadrature_points>
class source<specfem::enums::element::dimension::dim2,
             specfem::enums::element::medium::elastic, quadrature_points> {
public:
  KOKKOS_INLINE_FUNCTION virtual type_real
  eval_stf(const type_real &t) const = 0;

  KOKKOS_INLINE_FUNCTION virtual void
  compute_interaction(const int &xz, const type_real &stf_value,
                      type_real *accelx, type_real *accelz) const = 0;

  KOKKOS_INLINE_FUNCTION virtual void
  update_acceleration(const type_real &accelx, const type_real &accelz,
                      field_type field_dot_dot) const = 0;

  KOKKOS_INLINE_FUNCTION virtual int get_ispec() const = 0;
};

} // namespace sources
} // namespace impl
} // namespace domain
} // namespace specfem

#endif
