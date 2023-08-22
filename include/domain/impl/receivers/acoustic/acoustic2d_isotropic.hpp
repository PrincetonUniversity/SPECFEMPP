#ifndef DOMAIN_IMPL_RECEIVERS_ACOUSTIC2D_ISOTRPOIC_HPP_
#define DOMAIN_IMPL_RECEIVERS_ACOUSTIC2D_ISOTRPOIC_HPP_

#include "constants.hpp"
#include "domain/impl/receivers/acoustic/acoustic2d.hpp"
#include "domain/impl/receivers/receiver.hpp"
#include "specfem_enums.hpp"
#include <Kokkos_Core.hpp>

using sv_receiver_array_type =
    Kokkos::Subview<specfem::kokkos::DeviceView4d<type_real>, int,
                    std::remove_const_t<decltype(Kokkos::ALL)>,
                    std::remove_const_t<decltype(Kokkos::ALL)>,
                    std::remove_const_t<decltype(Kokkos::ALL)> >;

using sv_receiver_seismogram_type =
    Kokkos::Subview<specfem::kokkos::DeviceView4d<type_real>,
                    std::remove_const_t<decltype(Kokkos::ALL)>, int, int,
                    std::remove_const_t<decltype(Kokkos::ALL)> >;

using sv_receiver_field_type =
    Kokkos::Subview<specfem::kokkos::DeviceView6d<type_real>,
                    std::remove_const_t<decltype(Kokkos::ALL)>, int, int,
                    std::remove_const_t<decltype(Kokkos::ALL)>,
                    std::remove_const_t<decltype(Kokkos::ALL)>,
                    std::remove_const_t<decltype(Kokkos::ALL)> >;

namespace specfem {
namespace domain {
namespace impl {
namespace receivers {

template <int NGLL>
class receiver<
    specfem::enums::element::dimension::dim2,
    specfem::enums::element::medium::acoustic,
    specfem::enums::element::quadrature::static_quadrature_points<NGLL>,
    specfem::enums::element::property::isotropic>
    : public receiver<specfem::enums::element::dimension::dim2,
                      specfem::enums::element::medium::acoustic,
                      specfem::enums::element::quadrature::
                          static_quadrature_points<NGLL> > {

public:
  using dimension = specfem::enums::element::dimension::dim2;
  using medium = specfem::enums::element::medium::acoustic;
  using quadrature_points =
      specfem::enums::element::quadrature::static_quadrature_points<NGLL>;
  template <typename T>
  using ScratchViewType =
      typename specfem::enums::element::quadrature::static_quadrature_points<
          NGLL>::template ScratchViewType<T>;

  KOKKOS_FUNCTION
  receiver() = default;

  KOKKOS_FUNCTION
  receiver(const int ispec, const type_real sin_rec, const type_real cos_rec,
           const specfem::enums::seismogram::type seismogram,
           const sv_receiver_array_type receiver_array,
           const sv_receiver_seismogram_type receiver_seismogram,
           const specfem::compute::partial_derivatives &partial_derivatives,
           const specfem::compute::properties &properties,
           sv_receiver_field_type receiver_field);

  KOKKOS_INLINE_FUNCTION
  void get_field(const int xz, const int isig_step,
                 const ScratchViewType<type_real> field,
                 const ScratchViewType<type_real> field_dot,
                 const ScratchViewType<type_real> field_dot_dot,
                 const ScratchViewType<type_real> hprime_xx,
                 const ScratchViewType<type_real> hprime_zz) const override;

  KOKKOS_INLINE_FUNCTION
  void compute_seismogram_components(
      const int xz, const int isig_step,
      dimension::array_type<type_real> &l_seismogram_components) const override;

  KOKKOS_INLINE_FUNCTION
  void compute_seismogram(
      const int isig_step,
      const dimension::array_type<type_real> &seismogram_components) override;

  KOKKOS_INLINE_FUNCTION
  specfem::enums::seismogram::type get_seismogram_type() const override {
    return this->seismogram;
  }

  KOKKOS_INLINE_FUNCTION
  int get_ispec() const override { return this->ispec; }

private:
  const int ispec;
  const type_real sin_rec;
  const type_real cos_rec;
  const specfem::enums::seismogram::type seismogram;
  const sv_receiver_array_type receiver_array;
  const sv_receiver_seismogram_type receiver_seismogram;
  specfem::kokkos::DeviceView2d<type_real> xix;
  specfem::kokkos::DeviceView2d<type_real> gammax;
  specfem::kokkos::DeviceView2d<type_real> xiz;
  specfem::kokkos::DeviceView2d<type_real> gammaz;
  specfem::kokkos::DeviceView2d<type_real> rho_inverse;
  sv_receiver_field_type receiver_field;
};

} // namespace receivers
} // namespace impl
} // namespace domain
} // namespace specfem

#endif /* DOMAIN_IMPL_RECEIVERS_ACOUSTIC2D_ISOTRPOIC_HPP_ */
