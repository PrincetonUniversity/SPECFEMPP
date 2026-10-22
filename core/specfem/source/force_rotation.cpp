#include "specfem/source/force_rotation.hpp"

#include "specfem/coordinate_systems/spherical_basis.hpp"

specfem::sources::CartesianForce
specfem::sources::rotate_force_geographic_to_cartesian(
    double theta, double phi, const specfem::sources::GeographicForce &f) {

  const auto basis =
      specfem::coordinate_systems::local_geographic_basis(theta, phi);

  const double fx =
      f.east * basis.east[0] + f.north * basis.north[0] + f.up * basis.up[0];
  const double fy =
      f.east * basis.east[1] + f.north * basis.north[1] + f.up * basis.up[1];
  const double fz =
      f.east * basis.east[2] + f.north * basis.north[2] + f.up * basis.up[2];

  return { static_cast<type_real>(fx), static_cast<type_real>(fy),
           static_cast<type_real>(fz) };
}

specfem::sources::CartesianForce specfem::sources::geographic_to_enu(
    const specfem::sources::GeographicForce &f) {
  return { f.east, f.north, f.up };
}
