#pragma once

#include "element_types.hpp"
#include "enumerations/interface.hpp"
#include "mesh.hpp"
#include "mesh/mesh.hpp"
#include "receiver/interface.hpp"
#include <Kokkos_Core.hpp>
#include <memory>
#include <receiver/receiver.hpp>

namespace specfem::assembly {

/**
 * @brief Struct to store information related to the receivers
 *
 */

template <specfem::dimension::type DimensionTag> struct receivers;

} // namespace specfem::assembly

#include "receivers/dim2/receivers.hpp"
