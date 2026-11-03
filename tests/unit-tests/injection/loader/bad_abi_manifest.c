/* A plugin whose manifest claims an unsupported ABI version: the loader must
 * reject it before dlopen. */

#include "stub_table.h"

__attribute__((used, section(SPECFEM_MANIFEST_SECTION)))
SPECFEM_MANIFEST_RETAIN static const specfem_manifest bad_abi_manifest = {
  SPECFEM_MANIFEST_MAGIC_INITIALIZER,
  SPECFEM_MANIFEST_VERSION,
  99u,
  SPECFEM_PLUGIN_KIND_INJECTION,
  0u,
  "loader-bad-abi",
  "0.0.0"
};

STUB_TABLE(99u, "loader-bad-abi");

SPECFEM_EXPORT const specfem_injection_plugin_table *specfem_injection_plugin(void) {
  return &stub_table;
}
