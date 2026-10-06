#pragma once

/**
 * @brief Umbrella header for the @c specfem::injection component.
 *
 * Wavefield-injection methods for driving a local simulation with an
 * externally computed incident wavefield. Currently exposes the
 * frequency-wavenumber (FK) solver; future injection methods are added here.
 */

#include "specfem/injection/fk.hpp"
#include "specfem/injection/fk_provider.hpp"
#include "specfem/injection/injection_frame_buffer.hpp"
#include "specfem/injection/injection_provider.hpp"
