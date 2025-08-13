#include "specfem/jacobian.hpp"
#include "macros.hpp"
#include "specfem/shape_functions.hpp"

specfem::point::global_coordinates<specfem::dimension::type::dim2>
specfem::jacobian::compute_locations(
    const Kokkos::View<
        point::global_coordinates<specfem::dimension::type::dim2> *,
        Kokkos::HostSpace> &coorg,
    const int ngnod, const type_real xi, const type_real gamma) {

  auto shape2D = specfem::shape_function::shape_function(xi, gamma, ngnod);

  type_real xcor = 0.0;
  type_real zcor = 0.0;

  for (int in = 0; in < ngnod; in++) {
    xcor += shape2D[in] * coorg(in).x;
    zcor += shape2D[in] * coorg(in).z;
  }

  return { xcor, zcor };
}

specfem::point::jacobian_matrix<specfem::dimension::type::dim2, true, false>
specfem::jacobian::compute_jacobian(
    const Kokkos::View<
        point::global_coordinates<specfem::dimension::type::dim2> *,
        Kokkos::HostSpace> &coorg,
    const int ngnod, const std::vector<std::vector<type_real> > &dershape2D) {

  type_real xxi = 0.0;
  type_real zxi = 0.0;
  type_real xgamma = 0.0;
  type_real zgamma = 0.0;

  for (int in = 0; in < ngnod; in++) {
    xxi += dershape2D[0][in] * coorg(in).x;
    zxi += dershape2D[0][in] * coorg(in).z;
    xgamma += dershape2D[1][in] * coorg(in).x;
    zgamma += dershape2D[1][in] * coorg(in).z;
  }

  auto jacobian = xxi * zgamma - xgamma * zxi;

  type_real xix = zgamma / jacobian;
  type_real gammax = -zxi / jacobian;
  type_real xiz = -xgamma / jacobian;
  type_real gammaz = xxi / jacobian;

  return { xix, gammax, xiz, gammaz, jacobian };
}

specfem::point::jacobian_matrix<specfem::dimension::type::dim2, true, false>
specfem::jacobian::compute_jacobian(
    const Kokkos::View<
        point::global_coordinates<specfem::dimension::type::dim2> *,
        Kokkos::HostSpace> &coorg,
    const int ngnod, const type_real xi, const type_real gamma) {
  const auto dershape2D =
      specfem::shape_function::shape_function_derivatives(xi, gamma, ngnod);
  return specfem::jacobian::compute_jacobian(coorg, ngnod, dershape2D);
}
