#pragma once
#include "enumerations/interface.hpp"

/**
 * @namespace specfem::receivers
 * @brief Namespace for structures that handle receivers and seismic data
 * capture
 *
 * This namespace contains receiver classes that capture wavefield data at
 * specific locations during seismic simulations. The base class is @ref
 * specfem::receivers::receiver, with template specializations for different
 * spatial dimensions.
 *
 * <b>2D receiver implementations</b>
 * - @ref specfem::receivers::receiver<specfem::dimension::type::dim2>
 *
 * <b>3D receiver implementations</b>
 * - @ref specfem::receivers::receiver<specfem::dimension::type::dim3>
 *
 * Receivers are responsible for:
 * - Recording displacement, velocity, or acceleration at specific coordinates
 * - Interpolating wavefield values from the finite element mesh
 * - Managing station metadata (network name, station name, coordinates)
 * - Outputting seismograms for comparison with observed data
 *
 * See also
 * - @ref specfem::receivers::receiver
 */
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
