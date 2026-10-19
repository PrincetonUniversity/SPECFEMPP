#include "specfem/source/force_rotation.hpp"

#include "specfem/coordinate_systems/spherical_basis.hpp"

specfem::sources::CartesianForce
specfem::sources::rotate_force_geographic_to_cartesian(
    double theta, double phi, const specfem::sources::GeographicForce &f) {

  const auto basis =
      specfem::coordinate_systems::local_spherical_basis(theta, phi);

  // East = e_phi, North = -e_theta, Up = e_r.
  const double fx = f.east * basis.e_phi[0] - f.north * basis.e_theta[0] +
                    f.up * basis.e_r[0];
  const double fy = f.east * basis.e_phi[1] - f.north * basis.e_theta[1] +
                    f.up * basis.e_r[1];
  const double fz = f.east * basis.e_phi[2] - f.north * basis.e_theta[2] +
                    f.up * basis.e_r[2];

  return { static_cast<type_real>(fx), static_cast<type_real>(fy),
           static_cast<type_real>(fz) };
}

specfem::sources::CartesianForce specfem::sources::geographic_to_enu(
    const specfem::sources::GeographicForce &f) {
  return { f.east, f.north, f.up };
}
