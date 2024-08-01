#ifndef _ENUMERATION_BOUNDARY_HPP_
#define _ENUMERATION_BOUNDARY_HPP_

#include <stdexcept>

namespace specfem {
namespace element {
enum class boundary_tag {
  // primary boundaries
  none,                  ///< no boundary
  acoustic_free_surface, ///< free surface boundary for acoustic elements
  stacey,                ///< stacey boundary for elements

  // composite boundaries
  composite_stacey_dirichlet ///< composite boundary for acoustic elements
};

/**
 * @brief Container class to store boundary tags
 *
 * Rule of 5 needs to be implemented for this class. operator= is explicitly
 * deleted to avoid accidental assignment of boundary tags
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
  boundary_tag_container(){};

  /**
   * @brief Construct a new boundary tag container object
   *
   * Please use operator+= to update the boundary tag container
   *
   * @param tag boundary tag
   */
  boundary_tag_container &operator=(const boundary_tag &tag) = delete;

  /**
   * @brief Update boundary tag container with new tag
   *
   * This function checks if a boundary can be of composite type and returns the
   * correct tags
   *
   * @param rtag boundary tag to be added
   */
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
        throw std::runtime_error("Invalid boundary tag");
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
        throw std::runtime_error("Invalid boundary tag");
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
        throw std::runtime_error("Invalid boundary tag");
        break;
      }
      break;
    default:
      throw std::runtime_error("Invalid boundary tag");
      break;
    }

    return *this;
  }

  /**
   * @brief Check if boundary tag container specifies a specific boundary tag
   *
   * This function checks if a boundary container specifies a specific boundary
   * tag
   *
   * @param tag boundary tag to be checked
   * @return bool true if boundary container specifies the boundary tag
   */
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
      throw std::runtime_error("Invalid boundary tag");
      break;
    }
  }

  bool operator!=(const boundary_tag &tag) const {
    return !(this->operator==(tag));
  }

  bool operator==(const boundary_tag_container &rtag) const {
    return (rtag.tag == this->tag);
  }

  bool operator!=(const boundary_tag_container &rtag) const {
    return (rtag.tag != this->tag);
  }

private:
  boundary_tag tag = boundary_tag::none; ///< boundary tag
};
} // namespace element
} // namespace specfem

#endif /* _ENUMERATION_BOUNDARY_HPP_ */
