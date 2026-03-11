#pragma once

#include "specfem/enums.hpp"

namespace specfem::assembly {

template <specfem::element::dimension_tag DimensionTag,
          specfem::element::attenuation_tag AttenuationTag>
struct Attenuation;

} // namespace specfem::assembly

#include "specfem/assembly/attenuation/dim2/attenuation.hpp"
#include "specfem/assembly/attenuation/dim3/attenuation.hpp"
