#ifndef _DOMIAN_IMPL_RECEIVERS_ACOUSTIC2D_HPP_
#define _DOMIAN_IMPL_RECEIVERS_ACOUSTIC2D_HPP_

#include "constants.hpp"
#include "domain/impl/receivers/receiver.hpp"
#include "enumerations/interface.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace domain {
namespace impl {
namespace receivers {

template <class qp_type>
class receiver<specfem::enums::element::dimension::dim2,
               specfem::enums::element::medium::acoustic, qp_type> {
public:
  using dimension = specfem::enums::element::dimension::dim2;
  using medium_type = specfem::enums::element::medium::acoustic;
  using quadrature_points_type = qp_type;

  template <typename T, int N>
  using ScratchViewType =
      typename quadrature_points_type::template ScratchViewType<T, N>;

  KOKKOS_INLINE_FUNCTION virtual void get_field(
      const int xz, const int isig_step,
      const ScratchViewType<type_real, medium_type::components> field,
      const ScratchViewType<type_real, medium_type::components> field_dot,
      const ScratchViewType<type_real, medium_type::components> field_dot_dot,
      const ScratchViewType<type_real, 1> hprime_xx,
      const ScratchViewType<type_real, 1> hprime_zz) const {};
  KOKKOS_INLINE_FUNCTION virtual void
  compute_seismogram_components(const int xz, const int isig_step,
                                specfem::kokkos::array_type<type_real, 2>
                                    &l_seismogram_components) const {};
  KOKKOS_INLINE_FUNCTION virtual void compute_seismogram(
      const int isig_step,
      const specfem::kokkos::array_type<type_real, 2> &seismogram_components){};
  KOKKOS_INLINE_FUNCTION virtual specfem::enums::seismogram::type
  get_seismogram_type() const = 0;
  KOKKOS_INLINE_FUNCTION virtual int get_ispec() const = 0;
};

} // namespace receivers
} // namespace impl
} // namespace domain
} // namespace specfem

#endif /* _DOMIAN_IMPL_RECEIVERS_ACOUSTIC2D_HPP_ */
