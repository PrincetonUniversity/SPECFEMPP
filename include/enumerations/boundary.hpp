#pragma once

#include "medium.hpp"
#include <Kokkos_Core.hpp>
#include <stdexcept>

namespace specfem {
namespace element {

/**
 * @brief Container class to store boundary tags
 *
 * Boundary tag container is used to store boundary tag at a specific quadrature
 * point. The container-type is needed to handle composite boundaries.
 *
 *
 */
class boundary_tag_container {
public:
  /**
   * @brief Get the tags object
   *
   * @return std::vector<boundary_tag> vector of boundary tags
   */
  inline boundary_tag get_tag() const { return tag; }

  /**
   * @brief Construct a new boundary tag container object
   *
   */
  KOKKOS_INLINE_FUNCTION
  boundary_tag_container(){};

  /**
   * @brief Construct a new boundary tag container object
   *
   * Please use operator+= to update the boundary tag container
   *
   * @param tag boundary tag
   */
  KOKKOS_INLINE_FUNCTION
  boundary_tag_container &operator=(const boundary_tag &tag) = delete;

  /**
   * @brief Update boundary tag container with new tag
   *
   * @param rtag boundary tag to be added
   */
  KOKKOS_INLINE_FUNCTION
  boundary_tag_container &operator+=(const boundary_tag &rtag) {
    switch (rtag) {
    case boundary_tag::none:
      break;
    case boundary_tag::acoustic_free_surface:
      switch (this->tag) {
      case boundary_tag::none:
        this->tag = rtag;
        break;
      case boundary_tag::acoustic_free_surface:
      case boundary_tag::composite_stacey_dirichlet:
        break;
      case boundary_tag::stacey:
        this->tag = boundary_tag::composite_stacey_dirichlet;
        break;
      default:
        Kokkos::abort("Invalid boundary tag");
        break;
      }
      break;
    case boundary_tag::stacey:
      switch (this->tag) {
      case boundary_tag::none:
        this->tag = rtag;
        break;
      case boundary_tag::acoustic_free_surface:
        this->tag = boundary_tag::composite_stacey_dirichlet;
        break;
      case boundary_tag::stacey:
      case boundary_tag::composite_stacey_dirichlet:
        break;
      default:
        Kokkos::abort("Invalid boundary tag");
        break;
      }
      break;
    case boundary_tag::composite_stacey_dirichlet:
      switch (this->tag) {
      case boundary_tag::none:
        this->tag = rtag;
        break;
      case boundary_tag::acoustic_free_surface:
      case boundary_tag::stacey:
      case boundary_tag::composite_stacey_dirichlet:
        break;
      default:
        Kokkos::abort("Invalid boundary tag");
        break;
      }
      break;
    default:
      Kokkos::abort("Invalid boundary tag");
      break;
    }

    return *this;
  }

  /**
   * @brief Update boundary tag container with new tag
   *
   * @param rtag boundary tag container to be added
   */
  KOKKOS_INLINE_FUNCTION
  void operator+=(const boundary_tag_container &rtag) {
    this->operator+=(rtag.tag);
  }

  /**
   * @brief Check if boundary tag container is equivalent to a specific boundary
   * tag
   *
   * @param tag boundary tag to be checked
   * @return bool true if boundary container specifies the boundary tag
   */
  KOKKOS_INLINE_FUNCTION
  bool operator==(const boundary_tag &tag) const {
    switch (tag) {
    case boundary_tag::none:
      return (this->tag == boundary_tag::none);
      break;
    case boundary_tag::acoustic_free_surface:
      return (this->tag == boundary_tag::acoustic_free_surface ||
              this->tag == boundary_tag::composite_stacey_dirichlet);
      break;
    case boundary_tag::stacey:
      return (this->tag == boundary_tag::stacey ||
              this->tag == boundary_tag::composite_stacey_dirichlet);
      break;
    case boundary_tag::composite_stacey_dirichlet:
      return (this->tag == boundary_tag::composite_stacey_dirichlet);
      break;
    default:
      Kokkos::abort("Invalid boundary tag");
      break;
    }
  }

  /**
   * @brief Check if boundary tag container is not equivalent to a specific
   * boundary tag
   *
   * @param rtag boundary tag to be checked
   * @return bool true if boundary container is not equivalent to the boundary
   * tag
   */
  KOKKOS_INLINE_FUNCTION
  bool operator!=(const boundary_tag &tag) const {
    return !(this->operator==(tag));
  }

  /**
   * @brief Check if boundary tag container is equivalent to another boundary
   * tag container
   *
   * @param rtag boundary tag container to be checked
   * @return bool true if boundary container equivalent to the boundary tag
   */
  KOKKOS_INLINE_FUNCTION
  bool operator==(const boundary_tag_container &rtag) const {
    return (rtag.tag == this->tag);
  }

  /**
   * @brief Check if boundary tag container is not equivalent to another
   * boundary tag container
   *
   * @param rtag boundary tag container to be checked
   * @return bool true if boundary container is not equivalent to the boundary
   * tag
   */
  KOKKOS_INLINE_FUNCTION
  bool operator!=(const boundary_tag_container &rtag) const {
    return (rtag.tag != this->tag);
  }

private:
  boundary_tag tag = boundary_tag::none; ///< boundary tag
};
} // namespace element
} // namespace specfem
