#pragma once

namespace fk_injection {

/**
 * @brief Scalar type of the FK computations (CMake option
 * `FK_INJECTION_DOUBLE`).
 *
 * Independent of the ABI buffer dtype, which the host chooses; the plugin
 * adapter converts.
 */
#ifdef FK_INJECTION_DOUBLE_PRECISION
using type_real = double;
#else
using type_real = float;
#endif

} // namespace fk_injection
