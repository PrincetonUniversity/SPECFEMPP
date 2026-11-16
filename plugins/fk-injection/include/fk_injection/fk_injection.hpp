#pragma once

/**
 * @brief Umbrella header for the frequency-wavenumber (FK) injection solver.
 *
 * Pulls in the public input/result types of the FK solver. Solver-internal
 * drivers and per-medium operators live under @c src/impl and are not exposed
 * here.
 */

#include "fk_injection/config.hpp"
#include "fk_injection/eval_points.hpp"
#include "fk_injection/field_derivative.hpp"
#include "fk_injection/fk_result.hpp"
#include "fk_injection/incident_wave.hpp"
#include "fk_injection/layered_model.hpp"
#include "fk_injection/model_file_reader.hpp"
#include "fk_injection/solver.hpp"
#include "fk_injection/time_window.hpp"
#include "fk_injection/utilities/cubic_bspline.hpp"
#include "fk_injection/utilities/fft.hpp"
#include "fk_injection/utilities/register_array.hpp"
