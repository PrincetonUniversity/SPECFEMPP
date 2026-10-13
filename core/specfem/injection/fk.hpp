#pragma once

/**
 * @brief Umbrella header for the frequency-wavenumber (FK) injection solver.
 *
 * Pulls in the public input/result types of the FK solver. Solver-internal
 * drivers and per-medium operators live under @c fk/impl and are not exposed
 * here.
 */

#include "specfem/injection/fk/eval_points.hpp"
#include "specfem/injection/fk/field_derivative.hpp"
#include "specfem/injection/fk/fk_result.hpp"
#include "specfem/injection/fk/incident_wave.hpp"
#include "specfem/injection/fk/layered_model.hpp"
#include "specfem/injection/fk/time_window.hpp"
