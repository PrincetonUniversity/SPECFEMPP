// The plugin's only exported symbol. Kept apart from the implementation so the
// tests can link the implementation without defining the entry point.

#include "plane_wave_plugin.hpp"

SPECFEM_DEFINE_INJECTION_PLUGIN("plane-wave", "1.0.0",
                                plane_wave::PlaneWavePlugin)
