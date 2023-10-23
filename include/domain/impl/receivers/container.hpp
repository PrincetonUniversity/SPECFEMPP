#ifndef _RECEIVER_CONTAINER_HPP
#define _RECEIVER_CONTAINER_HPP

#include "enumerations/interface.hpp"

namespace specfem {
namespace domain {
namespace impl {
namespace receivers {

template <typename base_elemental_receiver> struct container {
public:
  base_elemental_receiver *receiver;

  KOKKOS_FUNCTION
  container() = default;

  KOKKOS_FUNCTION
  container(base_elemental_receiver *receiver) {
    this->receiver = receiver;
    return;
  }

  template <typename... Args>
  KOKKOS_INLINE_FUNCTION void get_field(Args... values) const {
    this->receiver->get_field(values...);
    return;
  }

  KOKKOS_INLINE_FUNCTION specfem::enums::seismogram::type
  get_seismogram_type() const {
    return this->receiver->get_seismogram_type();
  }

  template <typename... Args>
  KOKKOS_INLINE_FUNCTION void compute_seismogram(Args... values) const {
    this->receiver->compute_seismogram(values...);
    return;
  }

  template <typename... Args>
  KOKKOS_INLINE_FUNCTION void
  compute_seismogram_components(Args &&...values) const {
    this->receiver->compute_seismogram_components(
        std::forward<Args>(values)...);
    return;
  }

  KOKKOS_INLINE_FUNCTION int get_ispec() const {
    return this->receiver->get_ispec();
  }

  KOKKOS_FUNCTION
  ~container() = default;
};

} // namespace receivers
} // namespace impl
} // namespace domain
} // namespace specfem

#endif
