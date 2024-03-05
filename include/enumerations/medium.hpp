#ifndef _ENUMERATIONS_MEDIUM_HPP_
#define _ENUMERATIONS_MEDIUM_HPP_

#include "enumerations/dimension.hpp"
#include "specfem_enums.hpp"

namespace specfem {

namespace element {

constexpr int ntypes = 2; ///< Number of element types

enum class medium_tag {
  elastic,    ///< Elastic medium
  acoustic,   ///< Acoustic medium
  poroelastic ///< Poroelastic medium
};

enum class property_tag {
  isotropic, ///< Isotropic medium
  stacey     ///< Stacey medium
};
} // namespace element

namespace medium {

template <specfem::dimension::type Dimension,
          specfem::element::medium_tag MediumTag,
          specfem::element::property_tag PropertyTag =
              specfem::element::property_tag::isotropic>
class medium;

template <>
class medium<specfem::dimension::type::dim2,
             specfem::element::medium_tag::elastic,
             specfem::element::property_tag::isotropic> {
public:
  static constexpr auto dimension = specfem::dimension::type::dim2;
  static constexpr auto medium_tag = specfem::element::medium_tag::elastic;
  static constexpr auto property_tag =
      specfem::element::property_tag::isotropic;
  static constexpr int components = 2;
  static std::string to_string() { return "Elastic, Isotropic"; }
};

template <>
class medium<specfem::dimension::type::dim2,
             specfem::element::medium_tag::acoustic,
             specfem::element::property_tag::isotropic> {
public:
  static constexpr auto dimension = specfem::dimension::type::dim2;
  static constexpr auto medium_tag = specfem::element::medium_tag::acoustic;
  static constexpr auto property_tag =
      specfem::element::property_tag::isotropic;
  static constexpr int components = 1;
  static std::string to_string() { return "Acoustic, Isotropic"; }
};

} // namespace medium
} // namespace specfem

#endif /* _ENUMERATIONS_MEDIUM_HPP_ */
