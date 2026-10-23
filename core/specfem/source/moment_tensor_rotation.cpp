#include "specfem/source/moment_tensor_rotation.hpp"

#include "specfem/coordinate_systems/spherical_basis.hpp"

#include <array>

namespace specfem::sources_impl {

using Matrix3 = std::array<std::array<double, 3>, 3>;

// R * M * R^T for a symmetric 3x3 M, with R given by its columns. All math in
// double; the result is narrowed to type_real by the caller.
Matrix3 rotate_symmetric(const Matrix3 &r, const Matrix3 &m);

} // namespace specfem::sources_impl

specfem::sources_impl::Matrix3 specfem::sources_impl::rotate_symmetric(
    const specfem::sources_impl::Matrix3 &r,
    const specfem::sources_impl::Matrix3 &m) {

  // tmp = R * M
  Matrix3 tmp{};
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) {
      double sum = 0.0;
      for (int k = 0; k < 3; ++k)
        sum += r[i][k] * m[k][j];
      tmp[i][j] = sum;
    }

  // out = tmp * R^T
  Matrix3 out{};
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) {
      double sum = 0.0;
      for (int k = 0; k < 3; ++k)
        sum += tmp[i][k] * r[j][k];
      out[i][j] = sum;
    }

  return out;
}

specfem::sources::CartesianMomentTensor
specfem::sources::rotate_moment_tensor_geographic_to_cartesian(
    double theta, double phi,
    const specfem::sources::GeographicMomentTensor &m) {

  const auto basis =
      specfem::coordinate_systems::local_spherical_basis(theta, phi);

  // R = [ e_r | e_theta | e_phi ] (basis vectors as columns).
  const specfem::sources_impl::Matrix3 rotation = {
    { { basis.e_r[0], basis.e_theta[0], basis.e_phi[0] },
      { basis.e_r[1], basis.e_theta[1], basis.e_phi[1] },
      { basis.e_r[2], basis.e_theta[2], basis.e_phi[2] } }
  };

  // Symmetric tensor in (r, theta, phi) order.
  const specfem::sources_impl::Matrix3 spherical = { { { m.Mrr, m.Mrt, m.Mrp },
                                                       { m.Mrt, m.Mtt, m.Mtp },
                                                       { m.Mrp, m.Mtp,
                                                         m.Mpp } } };

  const auto c = specfem::sources_impl::rotate_symmetric(rotation, spherical);

  return { static_cast<type_real>(c[0][0]), static_cast<type_real>(c[1][1]),
           static_cast<type_real>(c[2][2]), static_cast<type_real>(c[0][1]),
           static_cast<type_real>(c[0][2]), static_cast<type_real>(c[1][2]) };
}

specfem::sources::CartesianMomentTensor specfem::sources::geographic_to_enu(
    const specfem::sources::GeographicMomentTensor &m) {
  return { m.Mpp, m.Mtt, m.Mrr, -m.Mtp, m.Mrp, -m.Mrt };
}
