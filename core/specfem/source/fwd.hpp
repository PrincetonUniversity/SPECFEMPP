#pragma once

#include "enumerations/interface.hpp"

namespace specfem::sources {

template <specfem::dimension::type DimensionTag> class source;

template <specfem::dimension::type DimensionTag> class tensor_source;

template <specfem::dimension::type DimensionTag> class vector_source;

template <specfem::dimension::type DimensionTag> struct moment_tensor;

template <specfem::dimension::type DimensionTag> class force;

template <specfem::dimension::type DimensionTag> class cosserat_force;

template <specfem::dimension::type DimensionTag> class adjoint_source;

template <specfem::dimension::type DimensionTag> class external;

} // namespace specfem::sources
