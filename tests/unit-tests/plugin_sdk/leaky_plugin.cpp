// Deliberately broken plugin: built without specfem_add_injection_plugin and
// exports a second symbol, so the export checker must reject it.

#include "specfem/plugin/injection_abi.h"

extern "C" SPECFEM_EXPORT int specfem_leaked_symbol() { return 1; }

static const specfem_injection_plugin_table leaky_table{};

extern "C" SPECFEM_EXPORT const specfem_injection_plugin_table *
specfem_injection_plugin(void) {
  return &leaky_table;
}
