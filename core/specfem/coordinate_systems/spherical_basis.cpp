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

specfem::coordinate_systems::GeographicBasis
specfem::coordinate_systems::local_geographic_basis(double theta, double phi) {
  const auto basis =
      specfem::coordinate_systems::local_spherical_basis(theta, phi);

  return { basis.e_phi,
           { -basis.e_theta[0], -basis.e_theta[1], -basis.e_theta[2] },
           basis.e_r };
}
