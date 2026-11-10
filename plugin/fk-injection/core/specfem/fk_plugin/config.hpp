#pragma once

namespace specfem {
namespace fk_plugin {

/**
 * @brief Scalar type of the FK computations (CMake option
 * `FK_PLUGIN_DOUBLE`).
 *
 * Independent of the ABI buffer dtype, which the host chooses; the plugin
 * adapter converts.
 */
#ifdef SPECFEM_FK_PLUGIN_DOUBLE
using type_real = double;
#else
using type_real = float;
#endif

} // namespace fk_plugin

namespace fk_plugin_impl {

using specfem::fk_plugin::type_real;

} // namespace fk_plugin_impl
} // namespace specfem
