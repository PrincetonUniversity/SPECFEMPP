/* A plugin whose manifest claims an unsupported ABI version: the loader must
 * reject it before dlopen. */

#include "stub_table.h"

__attribute__((used, section(SPECFEMPP_MANIFEST_SECTION)))
SPECFEMPP_MANIFEST_RETAIN static const specfempp_manifest bad_abi_manifest = {
  SPECFEMPP_MANIFEST_MAGIC_INITIALIZER,
  SPECFEMPP_MANIFEST_VERSION,
  99u,
  SPECFEMPP_PLUGIN_KIND_INJECTION,
  0u,
  "loader-bad-abi",
  "0.0.0"
};

STUB_TABLE(99u, "loader-bad-abi");

SPECFEMPP_EXPORT const specfempp_injection_table *specfempp_injection_plugin(void) {
  return &stub_table;
}
