/* A plugin whose manifest claims an unsupported ABI version: the loader must
 * reject it before dlopen. */

#include "stub_table.h"

__attribute__((used, section(SPFP_MANIFEST_SECTION)))
SPFP_MANIFEST_RETAIN static const spfp_manifest bad_abi_manifest = {
  SPFP_MANIFEST_MAGIC_INITIALIZER,
  SPFP_MANIFEST_VERSION,
  99u,
  SPFP_PLUGIN_KIND_INJECTION,
  0u,
  "loader-bad-abi",
  "0.0.0"
};

STUB_TABLE(99u, "loader-bad-abi");

SPFP_EXPORT const spfp_injection_plugin *specfempp_injection_plugin(void) {
  return &stub_table;
}
