// Deliberately broken plugin: built without specfempp_add_injection_plugin and
// exports a second symbol, so the export checker must reject it.

#include "specfem/plugin/injection_abi.h"

extern "C" SPECFEMPP_EXPORT int specfempp_leaked_symbol() { return 1; }

static const specfempp_injection_plugin_table leaky_table{};

extern "C" SPECFEMPP_EXPORT const specfempp_injection_plugin_table *
specfempp_injection_plugin(void) {
  return &leaky_table;
}
