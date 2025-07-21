#include "jacobian/interface.hpp"
#include "macros.hpp"
#include "specfem/shape_functions.hpp"

std::tuple<type_real, type_real> specfem::jacobian::compute_locations(
    const specfem::kokkos::HostTeam::member_type &teamMember,
    const specfem::kokkos::HostScratchView2d<type_real> s_coorg,
    const int ngnod, const type_real xi, const type_real gamma) {

  ASSERT(s_coorg.extent(0) == ndim, "Dimension mismatch");
  ASSERT(s_coorg.extent(1) == ngnod, "Number of nodes mismatch");

  auto shape2D = specfem::shape_function::shape_function(xi, gamma, ngnod);

  // FIXME:: Multi reduction is not yet implemented in kokkos
  // This is hacky way of doing this using double vector loops
  // Use multiple reducers once kokkos enables the feature
  return specfem::jacobian::compute_locations(teamMember, s_coorg, ngnod,
                                              shape2D);
}

std::tuple<type_real, type_real> specfem::jacobian::compute_locations(
    const specfem::kokkos::HostView2d<type_real> coorg, const int ngnod,
    const type_real xi, const type_real gamma) {

  ASSERT(coorg.extent(0) == ndim, "Dimension mismatch");
  ASSERT(coorg.extent(1) == ngnod, "Number of nodes mismatch");

  auto shape2D = specfem::shape_function::shape_function(xi, gamma, ngnod);

  return specfem::jacobian::compute_locations(coorg, ngnod, shape2D);
}

std::tuple<type_real, type_real> specfem::jacobian::compute_locations(
    const specfem::kokkos::HostTeam::member_type &teamMember,
    const specfem::kokkos::HostScratchView2d<type_real> s_coorg,
    const int ngnod, const std::vector<type_real> shape2D) {

  ASSERT(s_coorg.extent(0) == ndim, "Dimension mismatch");
  ASSERT(s_coorg.extent(1) == ngnod, "Number of nodes mismatch");
  ASSERT(shape2D.size() == ngnod, "Number of nodes mismatch");

  type_real xcor = 0.0;
  type_real ycor = 0.0;

  // FIXME:: Multi reduction is not yet implemented in kokkos
  // This is hacky way of doing this using double vector loops
  // Use multiple reducers once kokkos enables the feature

  for (int in = 0; in < ngnod; in++) {
    xcor += shape2D[in] * s_coorg(0, in);
    ycor += shape2D[in] * s_coorg(1, in);
  }

  return std::make_tuple(xcor, ycor);
}

std::tuple<type_real, type_real> specfem::jacobian::compute_locations(
    const specfem::kokkos::HostView2d<type_real> s_coorg, const int ngnod,
    const std::vector<type_real> shape2D) {

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

std::tuple<type_real, type_real, type_real, type_real>
specfem::jacobian::compute_jacobian_matrix(
    const specfem::kokkos::HostTeam::member_type &teamMember,
    const specfem::kokkos::HostScratchView2d<type_real> s_coorg,
    const int ngnod, const type_real xi, const type_real gamma) {

  ASSERT(s_coorg.extent(0) == ndim, "Dimension mismatch");
  ASSERT(s_coorg.extent(1) == ngnod, "Number of nodes mismatch");

  type_real xxi = 0.0;
  type_real zxi = 0.0;
  type_real xgamma = 0.0;
  type_real zgamma = 0.0;

  const auto dershape2D =
      specfem::shape_function::shape_function_derivatives(xi, gamma, ngnod);

  // FIXME:: Multi reduction is not yet implemented in kokkos
  // This is hacky way of doing this using double vector loops
  // Use multiple reducers once kokkos enables the feature

  Kokkos::parallel_reduce(
      Kokkos::ThreadVectorRange(teamMember, ngnod),
      [=](const int &in, type_real &update_xxi) {
        update_xxi += dershape2D[0][in] * s_coorg(0, in);
      },
      xxi);
  Kokkos::parallel_reduce(
      Kokkos::ThreadVectorRange(teamMember, ngnod),
      [=](const int &in, type_real &update_zxi) {
        update_zxi += dershape2D[0][in] * s_coorg(1, in);
      },
      zxi);
  Kokkos::parallel_reduce(
      Kokkos::ThreadVectorRange(teamMember, ngnod),
      [=](const int &in, type_real &update_xgamma) {
        update_xgamma += dershape2D[1][in] * s_coorg(0, in);
      },
      xgamma);
  Kokkos::parallel_reduce(
      Kokkos::ThreadVectorRange(teamMember, ngnod),
      [=](const int &in, type_real &update_zgamma) {
        update_zgamma += dershape2D[1][in] * s_coorg(1, in);
      },
      zgamma);

  return std::make_tuple(xxi, zxi, xgamma, zgamma);
}

std::tuple<type_real, type_real, type_real, type_real>
specfem::jacobian::compute_jacobian_matrix(
    const specfem::kokkos::HostTeam::member_type &teamMember,
    const specfem::kokkos::HostScratchView2d<type_real> s_coorg,
    const int ngnod, const specfem::kokkos::HostView2d<type_real> dershape2D) {

  ASSERT(s_coorg.extent(0) == ndim, "Dimension mismatch");
  ASSERT(s_coorg.extent(1) == ngnod, "Number of nodes mismatch");
  ASSERT(dershape2D.extent(0) == ndim, "Dimension mismatch");
  ASSERT(dershape2D.extent(1) == ngnod, "Number of nodes mismatch");

  type_real xxi = 0.0;
  type_real zxi = 0.0;
  type_real xgamma = 0.0;
  type_real zgamma = 0.0;

  // FIXME:: Multi reduction is not yet implemented in kokkos
  // This is hacky way of doing this using double vector loops
  // Use multiple reducers once kokkos enables the feature

  Kokkos::parallel_reduce(
      Kokkos::ThreadVectorRange(teamMember, ngnod),
      [=](const int &in, type_real &update_xxi) {
        update_xxi += dershape2D(0, in) * s_coorg(0, in);
      },
      xxi);
  Kokkos::parallel_reduce(
      Kokkos::ThreadVectorRange(teamMember, ngnod),
      [=](const int &in, type_real &update_zxi) {
        update_zxi += dershape2D(0, in) * s_coorg(1, in);
      },
      zxi);
  Kokkos::parallel_reduce(
      Kokkos::ThreadVectorRange(teamMember, ngnod),
      [=](const int &in, type_real &update_xgamma) {
        update_xgamma += dershape2D(1, in) * s_coorg(0, in);
      },
      xgamma);
  Kokkos::parallel_reduce(
      Kokkos::ThreadVectorRange(teamMember, ngnod),
      [=](const int &in, type_real &update_zgamma) {
        update_zgamma += dershape2D(1, in) * s_coorg(1, in);
      },
      zgamma);

  return std::make_tuple(xxi, zxi, xgamma, zgamma);
}

std::tuple<type_real, type_real, type_real, type_real>
specfem::jacobian::compute_jacobian_matrix(
    const specfem::kokkos::HostView2d<type_real> s_coorg, const int ngnod,
    const type_real xi, const type_real gamma) {

  ASSERT(s_coorg.extent(0) == ndim, "Dimension mismatch");
  ASSERT(s_coorg.extent(1) == ngnod, "Number of nodes mismatch");

  type_real xxi = 0.0;
  type_real zxi = 0.0;
  type_real xgamma = 0.0;
  type_real zgamma = 0.0;

  const auto dershape2D =
      specfem::shape_function::shape_function_derivatives(xi, gamma, ngnod);

  for (int in = 0; in < ngnod; in++) {
    xxi += dershape2D[0][in] * s_coorg(0, in);
    zxi += dershape2D[0][in] * s_coorg(1, in);
    xgamma += dershape2D[1][in] * s_coorg(0, in);
    zgamma += dershape2D[1][in] * s_coorg(1, in);
  }

  return std::make_tuple(xxi, zxi, xgamma, zgamma);
}

type_real specfem::jacobian::compute_jacobian(const type_real xxi,
                                              const type_real zxi,
                                              const type_real xgamma,
                                              const type_real zgamma) {
  return xxi * zgamma - xgamma * zxi;
}

type_real specfem::jacobian::compute_jacobian(
    const specfem::kokkos::HostTeam::member_type &teamMember,
    const specfem::kokkos::HostScratchView2d<type_real> s_coorg,
    const int ngnod, const type_real xi, const type_real gamma) {
  auto [xxi, zxi, xgamma, zgamma] = specfem::jacobian::compute_jacobian_matrix(
      teamMember, s_coorg, ngnod, xi, gamma);
  return specfem::jacobian::compute_jacobian(xxi, zxi, xgamma, zgamma);
};

type_real specfem::jacobian::compute_jacobian(
    const specfem::kokkos::HostTeam::member_type &teamMember,
    const specfem::kokkos::HostScratchView2d<type_real> s_coorg,
    const int ngnod, const specfem::kokkos::HostView2d<type_real> dershape2D) {
  auto [xxi, zxi, xgamma, zgamma] = specfem::jacobian::compute_jacobian_matrix(
      teamMember, s_coorg, ngnod, dershape2D);
  return specfem::jacobian::compute_jacobian(xxi, zxi, xgamma, zgamma);
};

std::tuple<type_real, type_real, type_real, type_real>
specfem::jacobian::compute_inverted_derivatives(
    const specfem::kokkos::HostTeam::member_type &teamMember,
    const specfem::kokkos::HostScratchView2d<type_real> s_coorg,
    const int ngnod, const type_real xi, const type_real gamma) {

  auto [xxi, zxi, xgamma, zgamma] = specfem::jacobian::compute_jacobian_matrix(
      teamMember, s_coorg, ngnod, xi, gamma);
  auto jacobian = specfem::jacobian::compute_jacobian(xxi, zxi, xgamma, zgamma);

  type_real xix = zgamma / jacobian;
  type_real gammax = -zxi / jacobian;
  type_real xiz = -xgamma / jacobian;
  type_real gammaz = xxi / jacobian;

  return std::make_tuple(xix, gammax, xiz, gammaz);
}

specfem::point::jacobian_matrix<specfem::dimension::type::dim2, true, false>
specfem::jacobian::compute_derivatives(
    const specfem::kokkos::HostTeam::member_type &teamMember,
    const specfem::kokkos::HostScratchView2d<type_real> s_coorg,
    const int ngnod, const specfem::kokkos::HostView2d<type_real> dershape2D) {

  auto [xxi, zxi, xgamma, zgamma] = specfem::jacobian::compute_jacobian_matrix(
      teamMember, s_coorg, ngnod, dershape2D);
  auto jacobian = specfem::jacobian::compute_jacobian(xxi, zxi, xgamma, zgamma);

  type_real xix = zgamma / jacobian;
  type_real gammax = -zxi / jacobian;
  type_real xiz = -xgamma / jacobian;
  type_real gammaz = xxi / jacobian;

  return specfem::point::jacobian_matrix<specfem::dimension::type::dim2, true,
                                         false>(xix, gammax, xiz, gammaz,
                                                jacobian);
}

std::tuple<type_real, type_real, type_real, type_real>
specfem::jacobian::compute_inverted_derivatives(
    const specfem::kokkos::HostTeam::member_type &teamMember,
    const specfem::kokkos::HostScratchView2d<type_real> s_coorg,
    const int ngnod, const specfem::kokkos::HostView2d<type_real> dershape2D) {
  auto [xxi, zxi, xgamma, zgamma] = specfem::jacobian::compute_jacobian_matrix(
      teamMember, s_coorg, ngnod, dershape2D);
  auto jacobian = specfem::jacobian::compute_jacobian(xxi, zxi, xgamma, zgamma);

  type_real xix = zgamma / jacobian;
  type_real gammax = -zxi / jacobian;
  type_real xiz = -xgamma / jacobian;
  type_real gammaz = xxi / jacobian;

  return std::make_tuple(xix, gammax, xiz, gammaz);
}

std::tuple<type_real, type_real, type_real, type_real>
specfem::jacobian::compute_inverted_derivatives(
    const specfem::kokkos::HostView2d<type_real> s_coorg, const int ngnod,
    const type_real xi, const type_real gamma) {
  auto [xxi, zxi, xgamma, zgamma] =
      specfem::jacobian::compute_jacobian_matrix(s_coorg, ngnod, xi, gamma);
  auto jacobian = specfem::jacobian::compute_jacobian(xxi, zxi, xgamma, zgamma);

  type_real xix = zgamma / jacobian;
  type_real gammax = -zxi / jacobian;
  type_real xiz = -xgamma / jacobian;
  type_real gammaz = xxi / jacobian;

  return std::make_tuple(xix, gammax, xiz, gammaz);
}
