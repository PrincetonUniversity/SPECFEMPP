#pragma once
#include "enumerations/interface.hpp"
namespace specfem::receivers {

/**
 * @brief Receiver Class
 *
 */
template <specfem::dimension::type DimensionTag> class receiver;

} // namespace specfem::receivers

#include "receiver/dim2/receiver.hpp"
