#pragma once

#include "fk_injection/incident_wave.hpp"
#include "fk_injection/layered_model.hpp"
#include "fk_injection/time_window.hpp"

#include <string>

namespace fk_injection {

/**
 * @brief Contents of an FK model file.
 *
 * @c wave.gaussian_half_duration is 0: the file has no such key, and SPECFEM3D
 * sets it to 1 / @c FREQUENCY_MAX, which the caller derives after any
 * overrides.
 */
struct FkModelFile {
  LayeredModel model;          ///< Layered velocity model
  IncidentWave wave;           ///< Incident wave
  TimeWindow window;           ///< Time window
  type_real frequency_max = 0; ///< @c FREQUENCY_MAX in Hz
  bool has_origin = false;     ///< Whether the file gives @c ORIGIN_WAVEFRONT;
                               ///< SPECFEM3D derives the origin when it doesn't
};

/**
 * @brief Read a FK model file and parse it into typed SPECFEM++ objects.
 *
 * Ports the keyword-driven format of @c ReadFKModelInput from the Westervelt
 * reference implementation (fk_model.cpp).  The file format uses one keyword
 * per line followed by its value(s).  Lines beginning with @c # and blank
 * lines are ignored; an unknown keyword or unreadable value is an error.
 *
 * Supported keywords:
 *
 * | Keyword             | Value(s)                          | Notes |
 * |---------------------|-----------------------------------|--------------------------------------|
 * | @c NLAYER           | integer                           | Must appear
 * before any @c LAYER.     | | @c LAYER            | i rho vp vs ztop | 1-based
 * index i; ztop in m.          | | @c INCIDENT_WAVE    | @c p or @c sv
 * (case-insensitive)  | Default: P.                          | | @c
 * BACK_AZIMUTH     | degrees                           | @f$\phi =
 * -\mathrm{baz} - 90°@f$    | | @c AZIMUTH          | degrees | @f$\phi = 90° -
 * \mathrm{az}@f$      | | @c TAKE_OFF         | degrees | Take-off angle from
 * vertical.        | | @c ORIGIN_WAVEFRONT | x y z | Origin coordinates in m. |
 * | @c ORIGIN_TIME      | seconds                           | Reference origin
 * time.               | | @c NSTEP            | integer | Simulation step
 * count.               | | @c deltat           | seconds | Simulation time
 * step.                | | @c FREQUENCY_MAX    | Hz | Maximum frequency of
 * interest.       | | @c FREQUENCY_SAMPLING | Hz                              |
 * FK storage sampling frequency.       | | @c TIME_WINDOW      | seconds | FK
 * time-window length.               | | @c AMPLITUDE        | scalar |
 * Plane-wave amplitude scale.          |
 *
 * Layer classification: a @c LAYER entry with @c vs < 1e-6 becomes an
 * @c AcousticLayer; otherwise it becomes an @c ElasticIsotropicLayer.
 * All acoustic (fluid) layers must form a contiguous block at the top of the
 * model (lowest layer indices).  Layer thicknesses are derived as
 * @f$ H_i = \mathrm{ztop}_i - \mathrm{ztop}_{i+1} @f$ (the last layer is the
 * half-space whose thickness is treated as zero by the FK propagator).
 *
 * Angles are converted from degrees to radians before storage.
 *
 * @param path Path to the FK model input file.
 * @return Model, wave, time window and whether the file gives the origin.
 * @throws std::runtime_error on file I/O failure or malformed input.
 */
FkModelFile read_fk_model_file(const std::string &path);

} // namespace fk_injection
