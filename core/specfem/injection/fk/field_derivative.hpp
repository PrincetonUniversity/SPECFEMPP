#pragma once

namespace specfem {
namespace injection {
namespace fk {

/**
 * @brief Kinematic quantity stored for the three vector field components.
 *
 * The FK recovery produces displacement \f$u\f$; the driver applies the
 * frequency-domain factor \f$(i\omega)^n\f$ before the inverse FFT to obtain
 * the requested time-derivative:
 * - @c displacement : \f$u\f$                        (needed by Method 2)
 * - @c velocity     : \f$(i\omega)\,u\f$             (needed by Method 1
 * Stacey)
 * - @c acceleration : \f$-\omega^2\,u = (i\omega)^2 u\f$ (needed by Method 2)
 *
 * The traction and pressure components are independent of this choice.
 */
enum class field_derivative { displacement, velocity, acceleration };

} // namespace fk
} // namespace injection
} // namespace specfem
