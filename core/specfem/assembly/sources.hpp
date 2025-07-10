#pragma once

#include "enumerations/interface.hpp"

namespace specfem::assembly {

template <specfem::dimension::type DimensionTag>
struct sources; ///< Forward declaration of sources class

} // namespace specfem::assembly

// Include template specializations
#include "sources/dim2/sources.hpp"
#include "sources/impl/source_array.hpp"
