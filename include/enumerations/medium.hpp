#ifndef _ENUMERATIONS_MEDIUM_HPP_
#define _ENUMERATIONS_MEDIUM_HPP_

#include "specfem_enums.hpp"

namespace specfem {

namespace element {

constexpr int ntypes = 2; ///< Number of element types

enum class medium_tag {
  elastic, ///< Elastic medium
  acoustic ///< Acoustic medium
};

enum class property_tag {
  isotropic, ///< Isotropic medium
  stacey     ///< Stacey medium
};
} // namespace element

namespace medium {

template <specfem::dimension::type dimension,
          specfem::element::medium_tag medium,
          specfem::element::property_tag property>
class medium {
public:
  static constexpr int components;
  static constexpr specfem::element::medium_tag value = medium;
  static constexpr specfem::element::property_tag property_value = property;
  static std::string to_string(){};
  using dimension = specfem::dimension::dimension<dimension>;
};

template <specfem::element::property_tag property>
static constexpr int
    medium<specfem::dimension::type::dim2,
           specfem::element::medium_tag::elastic, property>::components = 2;

template <specfem::element::property_tag property>
static constexpr int
    medium<specfem::dimension::type::dim2,
           specfem::element::medium_tag::acoustic, property>::components = 1;

template <>
std::string
medium<specfem::dimension::type::dim2, specfem::element::medium_tag::elastic,
       specfem::element::property_tag::isotropic>::to_string() {
  return "Elastic, Isotropic";
}

template <>
std::string
medium<specfem::dimension::type::dim2, specfem::element::medium_tag::acoustic,
       specfem::element::property_tag::isotropic>::to_string() {
  return "Acoustic, Isotropic";
}

} // namespace medium
} // namespace specfem

#endif /* _ENUMERATIONS_MEDIUM_HPP_ */
