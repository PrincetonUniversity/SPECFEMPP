#include "specfem/jacobian.hpp"
#include "macros.hpp"
#include "specfem/shape_functions.hpp"

specfem::point::global_coordinates<specfem::dimension::type::dim3>
specfem::jacobian::compute_locations(
    const Kokkos::View<
        point::global_coordinates<specfem::dimension::type::dim3> *,
        Kokkos::HostSpace> &coorg,
    const int ngnod, const type_real xi, const type_real eta,
    const type_real gamma) {

  auto shape3D = specfem::shape_function::shape_function(xi, eta, gamma, ngnod);

  type_real xcor = 0.0;
  type_real ycor = 0.0;
  type_real zcor = 0.0;

  for (int in = 0; in < ngnod; in++) {
    xcor += shape3D[in] * coorg(in).x;
    ycor += shape3D[in] * coorg(in).y;
    zcor += shape3D[in] * coorg(in).z;
  }

  return { xcor, ycor, zcor };
}

specfem::point::jacobian_matrix<specfem::dimension::type::dim3, true, false>
specfem::jacobian::compute_derivatives(
    const Kokkos::View<
        point::global_coordinates<specfem::dimension::type::dim3> *,
        Kokkos::HostSpace> &coorg,
    const int ngnod, const type_real xi, const type_real eta,
    const type_real gamma) {
  const auto dershape3D = specfem::shape_function::shape_function_derivatives(
      xi, eta, gamma, ngnod);

  type_real xxi = 0.0;
  type_real yxi = 0.0;
  type_real zxi = 0.0;
  type_real xeta = 0.0;
  type_real yeta = 0.0;
  type_real zeta = 0.0;
  type_real xgamma = 0.0;
  type_real ygamma = 0.0;
  type_real zgamma = 0.0;

  for (int in = 0; in < ngnod; in++) {
    xxi += dershape3D[0][in] * coorg(in).x;
    yxi += dershape3D[0][in] * coorg(in).y;
    zxi += dershape3D[0][in] * coorg(in).z;
    xeta += dershape3D[1][in] * coorg(in).x;
    yeta += dershape3D[1][in] * coorg(in).y;
    zeta += dershape3D[1][in] * coorg(in).z;
    xgamma += dershape3D[2][in] * coorg(in).x;
    ygamma += dershape3D[2][in] * coorg(in).y;
    zgamma += dershape3D[2][in] * coorg(in).z;
  }

  auto jacobian = xxi * (yeta * zgamma - ygamma * zeta) -
                  xeta * (yxi * zgamma - ygamma * zxi) +
                  xgamma * (yxi * zeta - yeta * zxi);

  type_real xix = (yeta * zgamma - ygamma * zeta) / jacobian;
  type_real xiy = (xgamma * zeta - xeta * zgamma) / jacobian;
  type_real xiz = (xeta * ygamma - xgamma * yeta) / jacobian;
  type_real etax = (ygamma * zxi - yxi * zgamma) / jacobian;
  type_real etay = (xxi * zgamma - xgamma * zxi) / jacobian;
  type_real etaz = (xgamma * yxi - xxi * ygamma) / jacobian;
  type_real gammax = (yxi * zeta - yeta * zxi) / jacobian;
  type_real gammay = (xeta * zxi - xxi * zeta) / jacobian;
  type_real gammaz = (xxi * yeta - xeta * yxi) / jacobian;

  return { xix, etax, gammax, xiy, etay, gammay, xiz, etaz, gammaz, jacobian };
}
