#pragma once

#include "element_types.hpp"
#include "enumerations/interface.hpp"
#include "execution/for_each_level.hpp"
#include "mesh.hpp"
#include "mesh/mesh.hpp"
#include "specfem/receivers.hpp"
#include <Kokkos_Core.hpp>
#include <memory>

namespace specfem::assembly {

/**
 * @brief Struct to store information related to the receivers
 *
 */

template <specfem::dimension::type DimensionTag> struct receivers;

} // namespace specfem::assembly

#include "receivers/dim2/receivers.hpp"
