/* A plugin whose manifest is valid but whose table reports another ABI
 * version: caught after dlopen, when the table is checked. */

#include "stub_table.h"

SPFP_DEFINE_INJECTION_MANIFEST("loader-table-mismatch", "0.0.0");

STUB_TABLE(2u, "loader-table-mismatch");

SPFP_EXPORT const spfp_injection_plugin *specfempp_injection_plugin(void) {
  return &stub_table;
}
