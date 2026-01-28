#pragma once

#include "enumerations/medium.hpp"
#include <type_traits>

namespace specfem::medium::impl {
template <specfem::dimension::type DimensionTag,
          specfem::element::medium_tag MediumTag,
          specfem::element::attenuation_tag AttenuationTag,
          typename Enable = void>
class AttenuationValues;

template <specfem::dimension::type DimensionTag,
          specfem::element::medium_tag MediumTag>
class AttenuationValues<DimensionTag, MediumTag,
                        specfem::element::attenuation_tag::none> {
public:
  AttenuationValues() = default;

  bool operator==(const AttenuationValues &other) const { return true; }

  std::string print() const { return ""; }
};
} // namespace specfem::medium::impl
