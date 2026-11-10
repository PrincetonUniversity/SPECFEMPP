/* A plugin with a valid entry point but no manifest: the loader must reject
 * it before dlopen. */

#include "stub_table.h"

STUB_TABLE(SPECFEM_INJECTION_ABI_VERSION, "loader-no-manifest");

SPECFEM_EXPORT const specfem_injection_plugin_table *specfem_injection_plugin(void) {
  return &stub_table;
}
