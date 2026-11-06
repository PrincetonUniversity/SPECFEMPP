#pragma once

/**
 * @brief Umbrella header for the frequency-wavenumber (FK) injection solver.
 *
 * Pulls in the public input types of the FK solver. The response, series table
 * and sampler live under @c src, their operators under @c src/impl.
 */

#include "fk_injection/config.hpp"
#include "fk_injection/incident_wave.hpp"
#include "fk_injection/layered_model.hpp"
#include "fk_injection/model_file_reader.hpp"
#include "fk_injection/reference_convention.hpp"
#include "fk_injection/time_window.hpp"
#include "fk_injection/utilities/complex_matrix.hpp"
#include "fk_injection/utilities/cubic_bspline.hpp"
#include "fk_injection/utilities/fft.hpp"
