#pragma once
#include "enumerations/interface.hpp"
namespace specfem::receivers {

/**
 * @brief Receiver Class
 *
 * This class is responsible for handling the reception of seismic data at a
 * specific station.
 *
 * @tparam DimensionTag The dimension tag (2D or 3D) for the receiver.
 *
 */
template <specfem::dimension::type DimensionTag> class receiver;

} // namespace specfem::receivers

#include "receivers/dim2/receiver.hpp"
#include "receivers/dim3/receiver.hpp"
