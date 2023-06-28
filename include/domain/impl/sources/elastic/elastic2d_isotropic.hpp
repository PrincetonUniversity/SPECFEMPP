#ifndef _DOMAIN_SOURCE_ELASTIC_ISOTROPIC2D_HPP
#define _DOMAIN_SOURCE_ELASTIC_ISOTROPIC2D_HPP

#include "compute/interface.hpp"
#include "domain/impl/sources/elastic/elastic2d.hpp"
#include "domain/impl/sources/source.hpp"
#include "kokkos_abstractions.h"
#include "source_time_function/source_time_function.hpp"
#include "specfem_enums.hpp"
#include <Kokkos_Core.hpp>

using field_type = Kokkos::Subview<
    specfem::kokkos::DeviceView2d<type_real, Kokkos::LayoutLeft>, int,
    std::remove_const_t<decltype(Kokkos::ALL)> >;

namespace specfem {
namespace domain {
namespace impl {
namespace sources {
template <int N>
class source<specfem::enums::element::dimension::dim2,
             specfem::enums::element::medium::elastic,
             specfem::enums::element::quadrature::static_quadrature_points<N>,
             specfem::enums::element::property::isotropic>
    : public source<
          specfem::enums::element::dimension::dim2,
          specfem::enums::element::medium::elastic,
          specfem::enums::element::quadrature::static_quadrature_points<N> > {

public:
  KOKKOS_FUNCTION source() = default;
  KOKKOS_FUNCTION source(const source &) = default;

  KOKKOS_FUNCTION source(const int &ispec,
                         specfem::kokkos::DeviceView3d<type_real> source_array,
                         specfem::forcing_function::stf *stf);

  KOKKOS_INLINE_FUNCTION void
  compute_interaction(const int &xz, const type_real &stf_value,
                      type_real *accelx, type_real *accelz) const override;

  KOKKOS_INLINE_FUNCTION type_real eval_stf(const type_real &t) const override {
    return stf->compute(t);
  }

  KOKKOS_INLINE_FUNCTION void
  update_acceleration(const type_real &accelx, const type_real &accelz,
                      field_type field_dot_dot) const override;

  KOKKOS_INLINE_FUNCTION int get_ispec() const override { return ispec; }

private:
  int ispec;
  specfem::forcing_function::stf *stf;
  specfem::kokkos::DeviceView3d<type_real> source_array;
};
} // namespace sources
} // namespace impl
} // namespace domain
} // namespace specfem

#endif
