#ifndef DOMAIN_IMPL_RECEIVERS_ACOUSTIC2D_ISOTRPOIC_HPP_
#define DOMAIN_IMPL_RECEIVERS_ACOUSTIC2D_ISOTRPOIC_HPP_

#include "constants.hpp"
#include "domain/impl/receivers/acoustic/acoustic2d.hpp"
#include "domain/impl/receivers/receiver.hpp"
#include "specfem_enums.hpp"
#include <Kokkos_Core.hpp>

// using sv_receiver_array_type =
//     Kokkos::Subview<specfem::kokkos::DeviceView4d<type_real>, int,
//                     std::remove_const_t<decltype(Kokkos::ALL)>,
//                     std::remove_const_t<decltype(Kokkos::ALL)>,
//                     std::remove_const_t<decltype(Kokkos::ALL)> >;

// using sv_receiver_seismogram_type =
//     Kokkos::Subview<specfem::kokkos::DeviceView4d<type_real>,
//                     std::remove_const_t<decltype(Kokkos::ALL)>, int, int,
//                     std::remove_const_t<decltype(Kokkos::ALL)> >;

// using sv_receiver_field_type =
//     Kokkos::Subview<specfem::kokkos::DeviceView6d<type_real>,
//                     std::remove_const_t<decltype(Kokkos::ALL)>,
//                     std::remove_const_t<decltype(Kokkos::ALL)>, int,
//                     std::remove_const_t<decltype(Kokkos::ALL)>,
//                     std::remove_const_t<decltype(Kokkos::ALL)>,
//                     std::remove_const_t<decltype(Kokkos::ALL)> >;

namespace specfem {
namespace domain {
namespace impl {
namespace receivers {

template <int NGLL>
class receiver<
    specfem::enums::element::dimension::dim2,
    specfem::enums::element::medium::acoustic,
    specfem::enums::element::quadrature::static_quadrature_points<NGLL>,
    specfem::enums::element::property::isotropic> {

public:
  using dimension = specfem::enums::element::dimension::dim2;
  using medium_type = specfem::enums::element::medium::acoustic;
  using quadrature_points_type =
      specfem::enums::element::quadrature::static_quadrature_points<NGLL>;
  template <typename T, int N>
  using ScratchViewType =
      typename quadrature_points_type::template ScratchViewType<T, N>;

  KOKKOS_FUNCTION
  receiver() = default;

  KOKKOS_FUNCTION
  receiver(const specfem::kokkos::DeviceView1d<type_real> sin_rec,
           const specfem::kokkos::DeviceView1d<type_real> cos_rec,
           const specfem::kokkos::DeviceView4d<type_real> receiver_array,
           const specfem::compute::partial_derivatives &partial_derivatives,
           const specfem::compute::properties &properties,
           specfem::kokkos::DeviceView6d<type_real> receiver_field);

  KOKKOS_INLINE_FUNCTION
  void get_field(
      const int &ireceiver, const int &iseis, const int &ispec,
      const specfem::enums::seismogram::type &siesmogram_type, const int &xz,
      const int &isig_step,
      const ScratchViewType<type_real, medium_type::components> field,
      const ScratchViewType<type_real, medium_type::components> field_dot,
      const ScratchViewType<type_real, medium_type::components> field_dot_dot,
      const ScratchViewType<type_real, 1> hprime_xx,
      const ScratchViewType<type_real, 1> hprime_zz) const;

  KOKKOS_INLINE_FUNCTION
  void compute_seismogram_components(
      const int &ireceiver, const int &iseis,
      const specfem::enums::seismogram::type &seismogram_type, const int &xz,
      const int &isig_step,
      dimension::array_type<type_real> &l_seismogram_components) const;

  KOKKOS_INLINE_FUNCTION
  void compute_seismogram(
      const int &ireceiver,
      const dimension::array_type<type_real> &seismogram_components,
      specfem::kokkos::DeviceView1d<type_real> receiver_seismogram) const;

private:
  specfem::kokkos::DeviceView1d<type_real> sin_rec;
  specfem::kokkos::DeviceView1d<type_real> cos_rec;
  specfem::kokkos::DeviceView4d<type_real> receiver_array;
  specfem::kokkos::DeviceView3d<type_real> xix;
  specfem::kokkos::DeviceView3d<type_real> gammax;
  specfem::kokkos::DeviceView3d<type_real> xiz;
  specfem::kokkos::DeviceView3d<type_real> gammaz;
  specfem::kokkos::DeviceView3d<type_real> rho_inverse;
  specfem::kokkos::DeviceView6d<type_real> receiver_field;
};

} // namespace receivers
} // namespace impl
} // namespace domain
} // namespace specfem

#endif /* DOMAIN_IMPL_RECEIVERS_ACOUSTIC2D_ISOTRPOIC_HPP_ */
