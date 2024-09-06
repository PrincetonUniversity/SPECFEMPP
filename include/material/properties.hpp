#pragma once

#include "enumerations/medium.hpp"

namespace specfem {
namespace material {
/**
 * @brief Material properties for a given medium and property
 *
 * @tparam MediumTag Medium tag for the material
 * @tparam PropertyTag Property tag for the material
 */
template <specfem::element::medium_tag MediumTag,
          specfem::element::property_tag PropertyTag>
class properties;
} // namespace material
} // namespace specfem
