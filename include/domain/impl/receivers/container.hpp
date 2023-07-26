#ifndef _RECEIVER_CONTAINER_HPP
#define _RECEIVER_CONTAINER_HPP

#include "specfem_enums.hpp"

namespace specfem {
namespace domain {
namespace impl {
namespace receivers {
template <typename base_elemental_receiver>
struct compute_seismogram_components_struct {
public:
  using medium = typename base_elemental_receiver::medium;
  base_elemental_receiver *receiver;
  // Have a look at
  // https://kokkos.github.io/kokkos-core-wiki/ProgrammingGuide/ParallelDispatch.html#reductions-with-an-array-of-results
  // reductions into an array
  using value_type = type_real[];

  using size_type = specfem::kokkos::DeviceView1d<type_real>::size_type;

  size_type value_count;

  KOKKOS_INLINE_FUNCTION compute_seismogram_components_struct() = default;

  KOKKOS_INLINE_FUNCTION
  compute_seismogram_components_struct(base_elemental_receiver *receiver)
      : value_count(2), receiver(receiver){};

  // Tell each thread how to initialize its reduction result.
  KOKKOS_INLINE_FUNCTION void init(value_type &update) const {
    for (size_type i = 0; i < value_count; ++i) {
      update[i] = 0.0;
    }
    return;
  }

  // "Join" intermediate results from different threads.
  // This should normally implement the same reduction
  // operation as operator().
  KOKKOS_INLINE_FUNCTION void join(value_type &update,
                                   const value_type &source) const {
    for (size_type i = 0; i < value_count; ++i) {
      update[i] += source[i];
    }
    return;
  }

  KOKKOS_INLINE_FUNCTION void operator()(const size_type xz,
                                         value_type update) const {
    this->receiver->compute_seismogram_components(xz, update);
    return;
  }
};

template <typename base_elemental_receiver> struct container {
public:
  base_elemental_receiver *receiver;
  compute_seismogram_components_struct<base_elemental_receiver>
      compute_seismogram_components;

  KOKKOS_FUNCTION
  container() = default;

  KOKKOS_FUNCTION
  container(base_elemental_receiver *receiver)
      : receiver(receiver),
        compute_seismogram_components(
            compute_seismogram_components_struct(receiver)) {}

  template <typename... Args>
  KOKKOS_INLINE_FUNCTION void get_field(Args &&...args) const {
    this->receiver->get_field(std::forward<Args>(args)...);
    return;
  }

  KOKKOS_INLINE_FUNCTION specfem::enums::seismogram::type
  get_seismogram_type() const {
    return this->receiver->get_seismogram_type();
  }

  template <typename... Args>
  KOKKOS_INLINE_FUNCTION void compute_seismogram(Args &&...args) const {
    this->receiver->compute_seismogram(std::forward<Args>(args)...);
    return;
  }

  KOKKOS_FUNCTION
  ~container() = default;
};

} // namespace receivers
} // namespace impl
} // namespace domain
} // namespace specfem

#endif
