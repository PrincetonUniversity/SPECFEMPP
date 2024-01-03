#ifndef _SOURCE_CONTAINER_HPP
#define _SOURCE_CONTAINER_HPP

namespace specfem {
namespace domain {
namespace impl {
namespace sources {
template <typename base_elemental_source> struct container {
public:
  base_elemental_source *source;

  KOKKOS_FUNCTION
  container() = default;

  KOKKOS_FUNCTION
  container(base_elemental_source *source) {
    this->source = source;
    return;
  }

  template <typename... Args>
  KOKKOS_INLINE_FUNCTION void compute_interaction(Args... values) const {
    this->source->compute_interaction(values...);
    return;
  }

  template <typename... Args>
  KOKKOS_INLINE_FUNCTION void update_acceleration(Args... values) const {
    this->source->update_acceleration(values...);
    return;
  }

  KOKKOS_INLINE_FUNCTION
  type_real eval_stf(const type_real &t) const {
    return this->source->eval_stf(t);
  }

  KOKKOS_INLINE_FUNCTION
  int get_ispec() const { return this->source->get_ispec(); }

  KOKKOS_FUNCTION
  ~container() = default;
};
} // namespace sources
} // namespace impl
} // namespace domain
} // namespace specfem

#endif
