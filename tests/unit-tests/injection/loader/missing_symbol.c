/* A plugin with a valid manifest that does not export the entry point. */

#include "specfem/plugin/injection_abi.h"

SPECFEM_DEFINE_INJECTION_MANIFEST("loader-missing-symbol", "0.0.0");

__attribute__((visibility("default"))) int specfem_not_the_entry_point(void) {
  return 0;
}
