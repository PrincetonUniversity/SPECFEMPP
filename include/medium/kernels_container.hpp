#pragma once

#include "enumerations/medium.hpp"
#include "kokkos_abstractions.h"
#include "point/coordinates.hpp"
#include "point/kernels.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace medium {

template <specfem::element::medium_tag MediumTag,
          specfem::element::property_tag PropertyTag>
class kernels_container;

} // namespace medium
} // namespace specfem

// Including the template specializations here so that kernels_container is
// an interface to the compute/kernels module
#include "dim2/acoustic/isotropic/kernels_container.hpp"
#include "dim2/elastic/anisotropic/kernels_container.hpp"
#include "dim2/elastic/isotropic/kernels_container.hpp"
