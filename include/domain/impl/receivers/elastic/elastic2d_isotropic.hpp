#ifndef _DOMAIN_IMPL_RECEIVERS_ELASTIC2D_ISOTROPIC_HPP_
#define _DOMAIN_IMPL_RECEIVERS_ELASTIC2D_ISOTROPIC_HPP_

#include "constants.hpp"
#include "domain/impl/receivers/elastic/elastic2d.hpp"
#include "domain/impl/receivers/receiver.hpp"
#include "kokkos_abstractions.h"
#include "quadrature/interface.hpp"
#include "specfem_enums.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace domain {
namespace impl {
namespace receivers {

template <int NGLL>
class receiver<
    specfem::enums::element::dimension::dim2,
    specfem::enums::element::medium::elastic,
    specfem::enums::element::quadrature::static_quadrature_points<NGLL>,
    specfem::enums::element::property::isotropic>
    : public receiver<specfem::enums::element::dimension::dim2,
                      specfem::enums::element::medium::elastic,
                      specfem::enums::element::quadrature::
                          static_quadrature_points<NGLL> > {
public:
  KOKKOS_FUNCTION receiver() = default;
  KOKKOS_FUNCTION
  receiver(const int irec, const int iseis, const int ispec,
           const type_real sin_rec, const type_real cos_rec,
           const specfem::enums::seismogram::type seismogram,
           const specfem::compute::receivers receivers,
           const specfem::kokkos::DeviceView3d<int> ibool,
           specfem::quadrature::quadrature *gllx,
           specfem::quadrature::quadrature *gllz);
  KOKKOS_INLINE_FUNCTION void
  get_field(const int xz, const specfem::kokkos::DeviceView2d<type_real> field,
            const specfem::kokkos::DeviceView2d<type_real> field_dot,
            const specfem::kokkos::DeviceView2d<type_real> field_dot_dot)
      const override;

  KOKKOS_INLINE_FUNCTION void compute_seismogram_components(
      const int xz, type_real (&l_seismogram_components)[2]) const override;
  KOKKOS_INLINE_FUNCTION void compute_seismogram(
      const int isig_step,
      const type_real (&seismogram_components)[2]) const override;

  KOKKOS_INLINE_FUNCTION specfem::enums::seismogram::type
  get_seismogram_type() const override {
    return this->seismogram;
  }

private:
  specfem::enums::seismogram::type seismogram;
  type_real sin_rec;
  type_real cos_rec;
  int ispec;
  specfem::kokkos::DeviceView3d<type_real> receiver_field;
  specfem::kokkos::DeviceView3d<type_real> receiver_array;
  specfem::kokkos::DeviceView2d<type_real> receiver_seismogram;
  specfem::kokkos::DeviceView2d<int> ibool;
  specfem::quadrature::quadrature *gllx;
  specfem::quadrature::quadrature *gllz;
};

} // namespace receivers
} // namespace impl
} // namespace domain
} // namespace specfem

#endif /* _DOMAIN_IMPL_RECEIVERS_ELASTIC2D_ISOTROPIC_HPP_ */
