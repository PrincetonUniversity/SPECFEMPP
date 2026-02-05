#pragma once

#include "specfem/element.hpp"
#include <type_traits>

namespace specfem::medium_container::impl {
template <specfem::element::dimension_tag DimensionTag,
          specfem::element::medium_tag MediumTag,
          specfem::element::attenuation_tag AttenuationTag,
          typename Enable = void>
class AttenuationValues;

template <specfem::element::dimension_tag DimensionTag,
          specfem::element::medium_tag MediumTag>
class AttenuationValues<DimensionTag, MediumTag,
                        specfem::element::attenuation_tag::none> {
public:
  AttenuationValues() = default;

  bool operator==(const AttenuationValues &other) const { return true; }

  std::string print() const { return ""; }
};
} // namespace specfem::medium_container::impl
