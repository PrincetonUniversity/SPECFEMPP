#include "../include/jacobian.h"
#include "../include/config.h"
#include "../include/kokkos_abstractions.h"
#include "../include/shape_functions.h"

std::tuple<type_real, type_real>
jacobian::compute_locations(const specfem::HostTeam::member_type &teamMember,
                            const specfem::HostScratchView2d<type_real> s_coorg,
                            const int ngnod, const type_real xi,
                            const type_real gamma) {

  assert(s_coorg.extent(0) == ndim);
  assert(s_coorg.extent(1) == ngnod);

  type_real xcor = 0.0;
  type_real ycor = 0.0;

  specfem::HostView1d<type_real> shape2D =
      shape_functions::define_shape_functions(xi, gamma, ngnod);

  // FIXME:: Multi reduction is not yet implemented in kokkos
  // This is hacky way of doing this using double vector loops
  // Use multiple reducers once kokkos enables the feature
  Kokkos::parallel_reduce(
      Kokkos::ThreadVectorRange(teamMember, ngnod),
      [&](const int &in, type_real &update_xcor) {
        update_xcor += shape2D(in) * s_coorg(0, in);
      },
      xcor);
  Kokkos::parallel_reduce(
      Kokkos::ThreadVectorRange(teamMember, ngnod),
      [&](const int &in, type_real &update_ycor) {
        update_ycor += shape2D(in) * s_coorg(1, in);
      },
      ycor);

  return std::make_tuple(xcor, ycor);
}

std::tuple<type_real, type_real>
jacobian::compute_locations(const specfem::HostTeam::member_type &teamMember,
                            const specfem::HostScratchView2d<type_real> s_coorg,
                            const int ngnod,
                            const specfem::HostView1d<type_real> shape2D) {

  assert(s_coorg.extent(0) == ndim);
  assert(s_coorg.extent(1) == ngnod);
  assert(shape2D.extent(0) == ngnod);

  type_real xcor = 0.0;
  type_real ycor = 0.0;

  // FIXME:: Multi reduction is not yet implemented in kokkos
  // This is hacky way of doing this using double vector loops
  // Use multiple reducers once kokkos enables the feature

  Kokkos::parallel_reduce(
      Kokkos::ThreadVectorRange(teamMember, ngnod),
      [&](const int &in, type_real &update_xcor) {
        update_xcor += shape2D(in) * s_coorg(0, in);
      },
      xcor);
  Kokkos::parallel_reduce(
      Kokkos::ThreadVectorRange(teamMember, ngnod),
      [&](const int &in, type_real &update_ycor) {
        update_ycor += shape2D(in) * s_coorg(1, in);
      },
      ycor);

  return std::make_tuple(xcor, ycor);
}

std::tuple<type_real, type_real, type_real, type_real>
jacobian::compute_partial_derivatives(
    const specfem::HostTeam::member_type &teamMember,
    const specfem::HostScratchView2d<type_real> s_coorg, const int ngnod,
    const type_real xi, const type_real gamma) {

  assert(s_coorg.extent(0) == ndim);
  assert(s_coorg.extent(1) == ngnod);

  type_real xxi = 0.0;
  type_real zxi = 0.0;
  type_real xgamma = 0.0;
  type_real zgamma = 0.0;

  specfem::HostView2d<type_real> dershape2D =
      shape_functions::define_shape_functions_derivatives(xi, gamma, ngnod);

  // FIXME:: Multi reduction is not yet implemented in kokkos
  // This is hacky way of doing this using double vector loops
  // Use multiple reducers once kokkos enables the feature

  Kokkos::parallel_reduce(
      Kokkos::ThreadVectorRange(teamMember, ngnod),
      [&](const int &in, type_real &update_xxi) {
        update_xxi += dershape2D(0, in) * s_coorg(0, in);
      },
      xxi);
  Kokkos::parallel_reduce(
      Kokkos::ThreadVectorRange(teamMember, ngnod),
      [&](const int &in, type_real &update_zxi) {
        update_zxi += dershape2D(0, in) * s_coorg(1, in);
      },
      zxi);
  Kokkos::parallel_reduce(
      Kokkos::ThreadVectorRange(teamMember, ngnod),
      [&](const int &in, type_real &update_xgamma) {
        update_xgamma += dershape2D(1, in) * s_coorg(0, in);
      },
      xgamma);
  Kokkos::parallel_reduce(
      Kokkos::ThreadVectorRange(teamMember, ngnod),
      [&](const int &in, type_real &update_zgamma) {
        update_zgamma += dershape2D(1, in) * s_coorg(1, in);
      },
      zgamma);

  return std::make_tuple(xxi, zxi, xgamma, zgamma);
}

std::tuple<type_real, type_real, type_real, type_real>
jacobian::compute_partial_derivatives(
    const specfem::HostTeam::member_type &teamMember,
    const specfem::HostScratchView2d<type_real> s_coorg, const int ngnod,
    const specfem::HostView2d<type_real> dershape2D) {

  assert(s_coorg.extent(0) == ndim);
  assert(s_coorg.extent(1) == ngnod);
  assert(dershape2D.extent(0) == ndim);
  assert(dershape2D.extent(1) == ngnod);

  type_real xxi = 0.0;
  type_real zxi = 0.0;
  type_real xgamma = 0.0;
  type_real zgamma = 0.0;

  // FIXME:: Multi reduction is not yet implemented in kokkos
  // This is hacky way of doing this using double vector loops
  // Use multiple reducers once kokkos enables the feature

  Kokkos::parallel_reduce(
      Kokkos::ThreadVectorRange(teamMember, ngnod),
      [&](const int &in, type_real &update_xxi) {
        update_xxi += dershape2D(0, in) * s_coorg(0, in);
      },
      xxi);
  Kokkos::parallel_reduce(
      Kokkos::ThreadVectorRange(teamMember, ngnod),
      [&](const int &in, type_real &update_zxi) {
        update_zxi += dershape2D(0, in) * s_coorg(1, in);
      },
      zxi);
  Kokkos::parallel_reduce(
      Kokkos::ThreadVectorRange(teamMember, ngnod),
      [&](const int &in, type_real &update_xgamma) {
        update_xgamma += dershape2D(1, in) * s_coorg(0, in);
      },
      xgamma);
  Kokkos::parallel_reduce(
      Kokkos::ThreadVectorRange(teamMember, ngnod),
      [&](const int &in, type_real &update_zgamma) {
        update_zgamma += dershape2D(1, in) * s_coorg(1, in);
      },
      zgamma);

  return std::make_tuple(xxi, zxi, xgamma, zgamma);
}

type_real jacobian::compute_jacobian(const type_real xxi, const type_real zxi,
                                     const type_real xgamma,
                                     const type_real zgamma) {
  return xxi * zgamma - xgamma * zxi;
}

type_real
jacobian::compute_jacobian(const specfem::HostTeam::member_type &teamMember,
                           const specfem::HostScratchView2d<type_real> s_coorg,
                           const int ngnod, const type_real xi,
                           const type_real gamma) {
  auto [xxi, zxi, xgamma, zgamma] = jacobian::compute_partial_derivatives(
      teamMember, s_coorg, ngnod, xi, gamma);
  return jacobian::compute_jacobian(xxi, zxi, xgamma, zgamma);
};

type_real
jacobian::compute_jacobian(const specfem::HostTeam::member_type &teamMember,
                           const specfem::HostScratchView2d<type_real> s_coorg,
                           const int ngnod,
                           const specfem::HostView2d<type_real> dershape2D) {
  auto [xxi, zxi, xgamma, zgamma] = jacobian::compute_partial_derivatives(
      teamMember, s_coorg, ngnod, dershape2D);
  return jacobian::compute_jacobian(xxi, zxi, xgamma, zgamma);
};
