#ifndef _ELEMENT_CONTAINER_HPP
#define _ELEMENT_CONTAINER_HPP

namespace specfem {
namespace domain {
namespace impl {
namespace elements {
template <typename base_element> struct container {
  base_element *element;

  KOKKOS_FUNCTION
  container() = default;

  KOKKOS_FUNCTION
  container(base_element *element) {
    this->element = element;
    return;
  }

  template <typename... Args>
  KOKKOS_FUNCTION void compute_gradient(Args... values) const {
    this->element->compute_gradient(values...);
    return;
  }

  template <typename... Args>
  KOKKOS_FUNCTION void compute_stress(Args... values) const {
    this->element->compute_stress(values...);
    return;
  }

  template <typename... Args>
  KOKKOS_FUNCTION void update_acceleration(Args... values) const {
    this->element->update_acceleration(values...);
    return;
  }

  KOKKOS_FUNCTION
  int get_ispec() const { return this->element->get_ispec(); }

  KOKKOS_FUNCTION
  ~container() = default;
};
} // namespace elements
} // namespace impl
} // namespace domain
} // namespace specfem

#endif
