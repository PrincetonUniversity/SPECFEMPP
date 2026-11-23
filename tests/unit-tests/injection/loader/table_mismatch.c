/* A plugin whose manifest is valid but whose table reports another ABI
 * version: caught after dlopen, when the table is checked. */

#include "stub_table.h"

SPECFEM_DEFINE_INJECTION_MANIFEST("loader-table-mismatch", "0.0.0");

STUB_TABLE(2u, "loader-table-mismatch");

SPECFEM_EXPORT const specfem_injection_plugin_table *specfem_injection_plugin(void) {
  return &stub_table;
}
