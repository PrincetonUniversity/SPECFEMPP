#pragma once

namespace fk_injection {

/**
 * @brief Kinematic quantity stored for the three vector field components.
 *
 * The FK recovery produces displacement \f$u\f$; the driver applies the
 * frequency-domain factor \f$(i\omega)^n\f$ before the inverse FFT to obtain
 * the requested time-derivative:
 * - @c displacement : \f$u\f$
 * - @c velocity     : \f$(i\omega)\,u\f$ (the SPECFEM3D @c Veloc_FK convention)
 *
 * The traction and potential-rate components are independent of this choice.
 */
enum class field_derivative { displacement, velocity };

} // namespace fk_injection
