// The plugin's only exported symbol. Kept apart from the implementation so the
// tests can link the implementation without defining the entry point.

#include "fk_injection_plugin.hpp"

SPECFEM_DEFINE_INJECTION_PLUGIN("fk", "1.0.0", fk_injection::FkInjectionPlugin)
