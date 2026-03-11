#pragma once

#include "specfem/enums.hpp"

namespace specfem::assembly::impl {

template <specfem::element::dimension_tag DimensionTag,
          specfem::element::medium_tag MediumTag,
          specfem::element::property_tag PropertyTag,
          specfem::element::attenuation_tag AttenuationTag>
struct attenuation_medium;

} // namespace specfem::assembly::impl

#include "specfem/assembly/attenuation/impl/attenuation_medium/dim2/constant_isotropic.tpp"
#include "specfem/assembly/attenuation/impl/attenuation_medium/dim3/constant_isotropic.tpp"
