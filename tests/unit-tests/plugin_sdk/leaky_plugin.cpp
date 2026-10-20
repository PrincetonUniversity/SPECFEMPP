// Deliberately broken plugin: built without specfempp_add_injection_plugin and
// exports a second symbol, so the export checker must reject it.

#include "specfem/plugin/injection_abi.h"

extern "C" SPFP_EXPORT int specfempp_leaked_symbol() { return 1; }

static const spfp_injection_plugin leaky_table{};

extern "C" SPFP_EXPORT const spfp_injection_plugin *
specfempp_injection_plugin(void) {
  return &leaky_table;
}
