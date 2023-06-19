#ifndef _ELEMENT_CONTAINER_HPP
#define _ELEMENT_CONTAINER_HPP

namespace specfem {
namespace domain {
namespace impl {
namespace elements {
template <typename base_element> struct container {
  base_element *element = NULL;

  container() = default;

  container(base_element *element) {
    this->element = element;
    return;
  }

  template <typename... Args> void compute_gradient(Args... values) {
    this->element->compute_gradient(values...);
    return;
  }

  template <typename... Args> void compute_stress(Args... values) {
    this->element->compute_stresss(values...);
    return;
  }

  template <typename... Args> void update_acceleration(Args... values) {
    this->element->update_acceleration(values...);
    return;
  }

  ~container() = default;
};
} // namespace elements
} // namespace impl
} // namespace domain
} // namespace specfem

#endif
