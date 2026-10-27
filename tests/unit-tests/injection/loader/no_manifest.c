/* A plugin with a valid entry point but no manifest: the loader must reject
 * it before dlopen. */

#include "stub_table.h"

STUB_TABLE(SPECFEMPP_INJECTION_ABI_VERSION, "loader-no-manifest");

SPECFEMPP_EXPORT const specfempp_injection_table *specfempp_injection_plugin(void) {
  return &stub_table;
}
