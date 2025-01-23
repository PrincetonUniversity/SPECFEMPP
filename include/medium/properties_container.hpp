#pragma once

#include "enumerations/medium.hpp"

namespace specfem {
namespace medium {

template <specfem::element::medium_tag type,
          specfem::element::property_tag property>
struct properties_container {
  static_assert("Material type not implemented");
};

} // namespace medium
} // namespace specfem

// Including the template specializations here so that properties_container is
// an interface to the compute module
#include "dim2/acoustic/isotropic/properties_container.hpp"
#include "dim2/elastic/anisotropic/properties_container.hpp"
#include "dim2/elastic/isotropic/properties_container.hpp"
