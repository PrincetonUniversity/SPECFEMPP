#ifndef _ELEMENT_CONTAINER_HPP
#define _ELEMENT_CONTAINER_HPP

namespace specfem {
namespace domain {
namespace impl {
namespace elements {
/**
 * @brief Struct to hold a pointer to an element
 *
 * Kokkos views have a syntantic problem when creating a view of pointers. For
 * example, let's say we have want to create a view of pointers to int. The
 * syntax would be: View<int* *>, this is ambigous since the compiler doesn't
 * know if the * is part of 2-dimensional view type or a pointer to int type. To
 * avoid this problem, we create a struct which holds a pointer to an element.
 * This struct is then used to create a view of structs. This allows us to
 * create a view of pointers to elements.
 *
 * @tparam base_element Base element class i.e. elastic, acoustic or poroelastic
 */
template <typename base_element> struct container {
  base_element *element; ///< Pointer to an element

  /**
   * @brief Default constructor
   *
   */
  KOKKOS_FUNCTION
  container() = default;

  /**
   * @brief Construct a new container object
   *
   * @param element Pointer to an element
   */
  KOKKOS_FUNCTION
  container(base_element *element) {
    this->element = element;
    return;
  }

  /**
   * @brief Wrapper to compute gradients method of the element
   *
   * @tparam Args Arguments to the compute_gradient method of the element
   * @param values Arguments to the compute_gradient method of the element
   */
  template <typename... Args>
  KOKKOS_INLINE_FUNCTION void compute_gradient(Args... values) const {
    this->element->compute_gradient(values...);
    return;
  }

  /**
   * @brief Wrapper to compute stresses method of the element
   *
   * @tparam Args Arguments to the compute_stress method of the element
   * @param values Arguments to the compute_stress method of the element
   */
  template <typename... Args>
  KOKKOS_INLINE_FUNCTION void compute_stress(Args... values) const {
    this->element->compute_stress(values...);
    return;
  }

  /**
   * @brief Wrapper to update acceleration method of the element
   *
   * @tparam Args Arguments to the update_acceleration method of the element
   * @param values Arguments to the update_acceleration method of the element
   */
  template <typename... Args>
  KOKKOS_INLINE_FUNCTION void update_acceleration(Args... values) const {
    this->element->update_acceleration(values...);
    return;
  }

  /**
   * @brief Wrapper to get_ispec method of the element
   *
   */
  KOKKOS_INLINE_FUNCTION
  int get_ispec() const { return this->element->get_ispec(); }

  /**
   * @brief Default destructor
   *
   */
  KOKKOS_FUNCTION
  ~container() = default;
};
} // namespace elements
} // namespace impl
} // namespace domain
} // namespace specfem

#endif
