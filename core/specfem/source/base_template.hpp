#pragma once
#include "specfem/element.hpp"

namespace specfem::sources {

// Vector sources
template <specfem::element::dimension_tag DimensionTag> class force;

template <specfem::element::dimension_tag DimensionTag> class adjoint_source;

template <specfem::element::dimension_tag DimensionTag> class cosserat_force;

template <specfem::element::dimension_tag DimensionTag> class external;

// Tensor sources
template <specfem::element::dimension_tag DimensionTag> class moment_tensor;

} // namespace specfem::sources
