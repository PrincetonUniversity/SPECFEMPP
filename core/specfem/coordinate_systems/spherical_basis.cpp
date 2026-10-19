#include "specfem/coordinate_systems/spherical_basis.hpp"

#include <cmath>

specfem::coordinate_systems::SphericalBasis
specfem::coordinate_systems::local_spherical_basis(double theta, double phi) {
  const double sin_theta = std::sin(theta);
  const double cos_theta = std::cos(theta);
  const double sin_phi = std::sin(phi);
  const double cos_phi = std::cos(phi);

  return { { sin_theta * cos_phi, sin_theta * sin_phi, cos_theta },
           { cos_theta * cos_phi, cos_theta * sin_phi, -sin_theta },
           { -sin_phi, cos_phi, 0.0 } };
}
