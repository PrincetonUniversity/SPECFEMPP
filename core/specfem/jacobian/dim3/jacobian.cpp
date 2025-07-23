#include "specfem/jacobian.hpp"
#include "macros.hpp"
#include "specfem/shape_functions.hpp"

namespace specfem::jacobian::impl {

std::tuple<type_real, type_real, type_real, type_real, type_real, type_real,
           type_real, type_real, type_real>
compute_jacobian_matrix3d(
    const specfem::kokkos::HostView2d<type_real> &s_coorg, const int ngnod,
    const std::vector<std::vector<type_real> > &dershape3D) {
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
    xxi += dershape3D[0][in] * s_coorg(0, in);
    yxi += dershape3D[0][in] * s_coorg(1, in);
    zxi += dershape3D[0][in] * s_coorg(2, in);
    xeta += dershape3D[1][in] * s_coorg(0, in);
    yeta += dershape3D[1][in] * s_coorg(1, in);
    zeta += dershape3D[1][in] * s_coorg(2, in);
    xgamma += dershape3D[2][in] * s_coorg(0, in);
    ygamma += dershape3D[2][in] * s_coorg(1, in);
    zgamma += dershape3D[2][in] * s_coorg(2, in);
  }

  return std::make_tuple(xxi, yxi, zxi, xeta, yeta, zeta, xgamma, ygamma,
                         zgamma);
}

type_real
compute_jacobian3d(const specfem::kokkos::HostView2d<type_real> &s_coorg,
                   const int ngnod,
                   const std::vector<std::vector<type_real> > &dershape3D) {
  auto [xxi, yxi, zxi, xeta, yeta, zeta, xgamma, ygamma, zgamma] =
      compute_jacobian_matrix3d(s_coorg, ngnod, dershape3D);
  return xxi * (yeta * zgamma - ygamma * zeta) -
         xeta * (yxi * zgamma - ygamma * zxi) +
         xgamma * (yxi * zeta - yeta * zxi);
};

} // namespace specfem::jacobian::impl

std::tuple<type_real, type_real, type_real>
specfem::jacobian::compute_locations(
    const specfem::kokkos::HostView2d<type_real> &s_coorg, const int ngnod,
    const type_real xi, const type_real eta, const type_real gamma) {

  auto shape3D = specfem::shape_function::shape_function(xi, eta, gamma, ngnod);

  ASSERT(s_coorg.extent(0) == ndim, "Dimension mismatch");
  ASSERT(s_coorg.extent(1) == ngnod, "Number of nodes mismatch");
  ASSERT(shape3D.size() == ngnod, "Number of nodes mismatch");

  type_real xcor = 0.0;
  type_real ycor = 0.0;
  type_real zcor = 0.0;

  for (int in = 0; in < ngnod; in++) {
    xcor += shape3D[in] * s_coorg(0, in);
    ycor += shape3D[in] * s_coorg(1, in);
    zcor += shape3D[in] * s_coorg(2, in);
  }

  return std::make_tuple(xcor, ycor, zcor);
}

std::tuple<type_real, type_real, type_real, type_real, type_real, type_real,
           type_real, type_real, type_real>
specfem::jacobian::compute_derivatives(
    const specfem::kokkos::HostView2d<type_real> &s_coorg, const int ngnod,
    const type_real xi, const type_real eta, const type_real gamma) {
  const auto dershape3D = specfem::shape_function::shape_function_derivatives(
      xi, eta, gamma, ngnod);

  ASSERT(s_coorg.extent(0) == ndim, "Dimension mismatch");
  ASSERT(s_coorg.extent(1) == ngnod, "Number of nodes mismatch");
  ASSERT(dershape3D.size() == ndim, "Dimension mismatch");
  ASSERT(dershape3D[0].size() == ngnod, "Number of nodes mismatch");

  auto [xxi, yxi, zxi, xeta, yeta, zeta, xgamma, ygamma, zgamma] =
      specfem::jacobian::impl::compute_jacobian_matrix3d(s_coorg, ngnod,
                                                         dershape3D);
  auto jacobian =
      specfem::jacobian::impl::compute_jacobian3d(s_coorg, ngnod, dershape3D);

  type_real xix = (yeta * zgamma - ygamma * zeta) / jacobian;
  type_real xiy = (xgamma * zeta - xeta * zgamma) / jacobian;
  type_real xiz = (xeta * ygamma - xgamma * yeta) / jacobian;
  type_real etax = (ygamma * zxi - yxi * zgamma) / jacobian;
  type_real etay = (xxi * zgamma - xgamma * zxi) / jacobian;
  type_real etaz = (xgamma * yxi - xxi * ygamma) / jacobian;
  type_real gammax = (yxi * zeta - yeta * zxi) / jacobian;
  type_real gammay = (xeta * zxi - xxi * zeta) / jacobian;
  type_real gammaz = (xxi * yeta - xeta * yxi) / jacobian;

  return std::make_tuple(xix, etax, gammax, xiy, etay, gammay, xiz, etaz,
                         gammaz);
}
