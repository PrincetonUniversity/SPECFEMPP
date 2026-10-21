/* A plugin with a valid entry point but no manifest: the loader must reject
 * it before dlopen. */

#include "stub_table.h"

STUB_TABLE(SPFP_INJECTION_ABI_VERSION, "loader-no-manifest");

SPFP_EXPORT const spfp_injection_plugin *specfempp_injection_plugin(void) {
  return &stub_table;
}
