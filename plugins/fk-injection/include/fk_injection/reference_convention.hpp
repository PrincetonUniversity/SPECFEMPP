#pragma once

namespace fk_injection {

/**
 * @brief How the FK synthesis treats the points where it deliberately differs
 *        from SPECFEM3D's FK routine.
 *
 * - @c exact : the inverse FFT normalizes by the resampled step @c dt_min, the
 *   frequency grid is @f$ 1/(N \Delta t_\text{min}) @f$, each point's delay
 *   is the exact plane-wave delay, and the window lasts until a fluid's
 *   reverberation decays.
 * - @c specfem3d : reproduces SPECFEM3D for comparison against it. The inverse
 *   FFT normalizes by the SEM step, so amplitudes are @f$ N_P @f$ times larger,
 *   its frequency grid stretches all times by @f$ N/(N-1) @f$, the origin time
 *   is truncated to whole samples, and the window has no reverberation floor.
 */
enum class ReferenceConvention { exact, specfem3d };

} // namespace fk_injection
