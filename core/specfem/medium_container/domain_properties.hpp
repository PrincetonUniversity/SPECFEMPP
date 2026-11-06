#pragma once

#include "specfem/medium/dim2/acoustic/isotropic/domain_properties.hpp"
#include "specfem/medium/dim2/elastic/anisotropic/domain_properties.hpp"
#include "specfem/medium/dim2/elastic/isotropic/domain_properties.hpp"
#include "specfem/medium/dim2/elastic/isotropic_cosserat/domain_properties.hpp"
#include "specfem/medium/dim2/poroelastic/isotropic/domain_properties.hpp"

// dim3 acoustic and elastic isotropic containers use the DimensionTag-generic
// dim2 definitions above. Only the genuinely 3D anisotropic parameterization
// needs its own specialization.
#include "specfem/medium/dim3/elastic/anisotropic/domain_properties.hpp"
