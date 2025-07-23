#include "specfem/jacobian.hpp"
#include "macros.hpp"
#include "specfem/shape_functions.hpp"

namespace specfem::jacobian::impl {

std::tuple<type_real, type_real, type_real, type_real> compute_jacobian_matrix(
    const specfem::kokkos::HostView2d<type_real> &s_coorg, const int ngnod,
    const std::vector<std::vector<type_real> > &dershape2D) {
  type_real xxi = 0.0;
  type_real zxi = 0.0;
  type_real xgamma = 0.0;
  type_real zgamma = 0.0;

  for (int in = 0; in < ngnod; in++) {
    xxi += dershape2D[0][in] * s_coorg(0, in);
    zxi += dershape2D[0][in] * s_coorg(1, in);
    xgamma += dershape2D[1][in] * s_coorg(0, in);
    zgamma += dershape2D[1][in] * s_coorg(1, in);
  }

  return std::make_tuple(xxi, zxi, xgamma, zgamma);
}

type_real
compute_jacobian(const specfem::kokkos::HostView2d<type_real> &s_coorg,
                 const int ngnod,
                 const std::vector<std::vector<type_real> > &dershape2D) {
  const auto [xxi, zxi, xgamma, zgamma] =
      compute_jacobian_matrix(s_coorg, ngnod, dershape2D);
  return xxi * zgamma - xgamma * zxi;
};

} // namespace specfem::jacobian::impl

std::tuple<type_real, type_real> specfem::jacobian::compute_locations(
    const specfem::kokkos::HostView2d<type_real> &s_coorg, const int ngnod,
    const std::vector<type_real> &shape2D) {

  ASSERT(s_coorg.extent(0) == ndim, "Dimension mismatch");
  ASSERT(s_coorg.extent(1) == ngnod, "Number of nodes mismatch");
  ASSERT(shape2D.size() == ngnod, "Number of nodes mismatch");

  type_real xcor = 0.0;
  type_real ycor = 0.0;

  for (int in = 0; in < ngnod; in++) {
    xcor += shape2D[in] * s_coorg(0, in);
    ycor += shape2D[in] * s_coorg(1, in);
  }

  return std::make_tuple(xcor, ycor);
}

std::tuple<type_real, type_real> specfem::jacobian::compute_locations(
    const specfem::kokkos::HostView2d<type_real> &s_coorg, const int ngnod,
    const type_real xi, const type_real gamma) {

  auto shape2D = specfem::shape_function::shape_function(xi, gamma, ngnod);

  return specfem::jacobian::compute_locations(s_coorg, ngnod, shape2D);
}

std::tuple<type_real, type_real, type_real, type_real, type_real>
specfem::jacobian::compute_derivatives(
    const specfem::kokkos::HostView2d<type_real> &s_coorg, const int ngnod,
    const std::vector<std::vector<type_real> > &dershape2D) {

  ASSERT(s_coorg.extent(0) == ndim, "Dimension mismatch");
  ASSERT(s_coorg.extent(1) == ngnod, "Number of nodes mismatch");
  ASSERT(dershape2D.size() == ndim, "Dimension mismatch");
  ASSERT(dershape2D[0].size() == ngnod, "Number of nodes mismatch");

  auto [xxi, zxi, xgamma, zgamma] =
      impl::compute_jacobian_matrix(s_coorg, ngnod, dershape2D);
  auto jacobian = impl::compute_jacobian(s_coorg, ngnod, dershape2D);

  type_real xix = zgamma / jacobian;
  type_real gammax = -zxi / jacobian;
  type_real xiz = -xgamma / jacobian;
  type_real gammaz = xxi / jacobian;

  return std::make_tuple(xix, gammax, xiz, gammaz, jacobian);
}

std::tuple<type_real, type_real, type_real, type_real, type_real>
specfem::jacobian::compute_derivatives(
    const specfem::kokkos::HostView2d<type_real> &s_coorg, const int ngnod,
    const type_real xi, const type_real gamma) {
  const auto dershape2D =
      specfem::shape_function::shape_function_derivatives(xi, gamma, ngnod);
  return specfem::jacobian::compute_derivatives(s_coorg, ngnod, dershape2D);
}
