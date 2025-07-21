#include "specfem/jacobian.hpp"
#include "macros.hpp"
#include "specfem/shape_functions.hpp"

std::tuple<type_real, type_real> specfem::jacobian::compute_locations(
    const specfem::kokkos::HostTeam::member_type &teamMember,
    const Kokkos::View<type_real ***, Kokkos::LayoutLeft, HostScratchSpace,
                       Kokkos::MemoryTraits<Kokkos::Unmanaged> > &s_coorg,
    const int ngnod, const type_real xi, const type_real eta,
    const type_real gamma) {

  ASSERT(s_coorg.extent(0) == ndim, "Dimension mismatch");
  ASSERT(s_coorg.extent(1) == ngnod, "Number of nodes mismatch");

  auto shape3D = specfem::shape_function::shape_function(xi, eta, gamma, ngnod);

  // FIXME:: Multi reduction is not yet implemented in kokkos
  // This is hacky way of doing this using double vector loops
  // Use multiple reducers once kokkos enables the feature
  return specfem::jacobian::compute_locations(teamMember, s_coorg, ngnod,
                                              shape3D);
}

std::tuple<type_real, type_real> specfem::jacobian::compute_locations(
    const Kokkos::View<type_real ***, Kokkos::LayoutLeft, HostSpace> &coorg,
    const int ngnod, const type_real xi, const type_real eta,
    const type_real gamma) {

  ASSERT(coorg.extent(0) == ndim, "Dimension mismatch");
  ASSERT(coorg.extent(1) == ngnod, "Number of nodes mismatch");

  auto shape3D = specfem::shape_function::shape_function(xi, eta, gamma, ngnod);

  return specfem::jacobian::compute_locations(coorg, ngnod, shape3D);
}

std::tuple<type_real, type_real, type_real>
specfem::jacobian::compute_locations(
    const specfem::kokkos::HostTeam::member_type &teamMember,
    const Kokkos::View<type_real ***, Kokkos::LayoutLeft, HostScratchSpace,
                       Kokkos::MemoryTraits<Kokkos::Unmanaged> > &s_coorg,
    const int ngnod, const std::vector<type_real> &shape3D) {

  ASSERT(s_coorg.extent(0) == ndim, "Dimension mismatch");
  ASSERT(s_coorg.extent(1) == ngnod, "Number of nodes mismatch");
  ASSERT(shape3D.size() == ngnod, "Number of nodes mismatch");

  type_real xcor = 0.0;
  type_real ycor = 0.0;
  type_real zcor = 0.0;

  // FIXME:: Multi reduction is not yet implemented in kokkos
  // This is hacky way of doing this using double vector loops
  // Use multiple reducers once kokkos enables the feature

  for (int in = 0; in < ngnod; in++) {
    xcor += shape3D[in] * s_coorg(0, in);
    ycor += shape3D[in] * s_coorg(1, in);
    zcor += shape3D[in] * s_coorg(2, in);
  }

  return std::make_tuple(xcor, ycor, zcor);
}

std::tuple<type_real, type_real, type_real>
specfem::jacobian::compute_locations(
    const Kokkos::View<type_real ***, Kokkos::LayoutLeft, HostSpace> &s_coorg,
    const int ngnod, const std::vector<type_real> &shape3D) {

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
specfem::jacobian::compute_jacobian_matrix(
    const specfem::kokkos::HostTeam::member_type &teamMember,
    const Kokkos::View<type_real ***, Kokkos::LayoutLeft, HostScratchSpace,
                       Kokkos::MemoryTraits<Kokkos::Unmanaged> > &s_coorg,
    const int ngnod, const type_real xi, const type_real eta,
    const type_real gamma) {

  ASSERT(s_coorg.extent(0) == ndim, "Dimension mismatch");
  ASSERT(s_coorg.extent(1) == ngnod, "Number of nodes mismatch");

  type_real xxi = 0.0;
  type_real yxi = 0.0;
  type_real zxi = 0.0;
  type_real xeta = 0.0;
  type_real yeta = 0.0;
  type_real zeta = 0.0;
  type_real xgamma = 0.0;
  type_real ygamma = 0.0;
  type_real zgamma = 0.0;

  const auto dershape3D = specfem::shape_function::shape_function_derivatives(
      xi, eta, gamma, ngnod);

  // FIXME:: Multi reduction is not yet implemented in kokkos
  // This is hacky way of doing this using double vector loops
  // Use multiple reducers once kokkos enables the feature

  Kokkos::parallel_reduce(
      Kokkos::ThreadVectorRange(teamMember, ngnod),
      [=](const int &in, type_real &update_xxi) {
        update_xxi += dershape3D[0][in] * s_coorg(0, in);
      },
      xxi);
  Kokkos::parallel_reduce(
      Kokkos::ThreadVectorRange(teamMember, ngnod),
      [=](const int &in, type_real &update_yxi) {
        update_yxi += dershape3D[0][in] * s_coorg(1, in);
      },
      yxi);
  Kokkos::parallel_reduce(
      Kokkos::ThreadVectorRange(teamMember, ngnod),
      [=](const int &in, type_real &update_zxi) {
        update_zxi += dershape3D[0][in] * s_coorg(2, in);
      },
      zxi);

  Kokkos::parallel_reduce(
      Kokkos::ThreadVectorRange(teamMember, ngnod),
      [=](const int &in, type_real &update_xeta) {
        update_xeta += dershape3D[1][in] * s_coorg(0, in);
      },
      xeta);
  Kokkos::parallel_reduce(
      Kokkos::ThreadVectorRange(teamMember, ngnod),
      [=](const int &in, type_real &update_yeta) {
        update_yeta += dershape3D[1][in] * s_coorg(1, in);
      },
      yeta);
  Kokkos::parallel_reduce(
      Kokkos::ThreadVectorRange(teamMember, ngnod),
      [=](const int &in, type_real &update_zeta) {
        update_zeta += dershape3D[1][in] * s_coorg(2, in);
      },
      zeta);

  Kokkos::parallel_reduce(
      Kokkos::ThreadVectorRange(teamMember, ngnod),
      [=](const int &in, type_real &update_xgamma) {
        update_xgamma += dershape3D[2][in] * s_coorg(0, in);
      },
      xgamma);
  Kokkos::parallel_reduce(
      Kokkos::ThreadVectorRange(teamMember, ngnod),
      [=](const int &in, type_real &update_ygamma) {
        update_ygamma += dershape3D[2][in] * s_coorg(1, in);
      },
      ygamma);
  Kokkos::parallel_reduce(
      Kokkos::ThreadVectorRange(teamMember, ngnod),
      [=](const int &in, type_real &update_zgamma) {
        update_zgamma += dershape3D[2][in] * s_coorg(2, in);
      },
      zgamma);

  return std::make_tuple(xxi, yxi, zxi, xeta, yeta, zeta, xgamma, ygamma,
                         zgamma);
}

std::tuple<type_real, type_real, type_real, type_real, type_real, type_real,
           type_real, type_real, type_real>
specfem::jacobian::compute_jacobian_matrix(
    const specfem::kokkos::HostTeam::member_type &teamMember,
    const Kokkos::View<type_real ***, Kokkos::LayoutLeft, HostScratchSpace,
                       Kokkos::MemoryTraits<Kokkos::Unmanaged> > &s_coorg,
    const int ngnod,
    const Kokkos::View<type_real ***, Kokkos::LayoutLeft, HostSpace>
        &dershape2D) {

  ASSERT(s_coorg.extent(0) == ndim, "Dimension mismatch");
  ASSERT(s_coorg.extent(1) == ngnod, "Number of nodes mismatch");
  ASSERT(dershape2D.extent(0) == ndim, "Dimension mismatch");
  ASSERT(dershape2D.extent(1) == ngnod, "Number of nodes mismatch");

  type_real xxi = 0.0;
  type_real yxi = 0.0;
  type_real zxi = 0.0;
  type_real xeta = 0.0;
  type_real yeta = 0.0;
  type_real zeta = 0.0;
  type_real xgamma = 0.0;
  type_real ygamma = 0.0;
  type_real zgamma = 0.0;

  // FIXME:: Multi reduction is not yet implemented in kokkos
  // This is hacky way of doing this using double vector loops
  // Use multiple reducers once kokkos enables the feature

  Kokkos::parallel_reduce(
      Kokkos::ThreadVectorRange(teamMember, ngnod),
      [=](const int &in, type_real &update_xxi) {
        update_xxi += dershape3D[0][in] * s_coorg(0, in);
      },
      xxi);
  Kokkos::parallel_reduce(
      Kokkos::ThreadVectorRange(teamMember, ngnod),
      [=](const int &in, type_real &update_yxi) {
        update_yxi += dershape3D[0][in] * s_coorg(1, in);
      },
      yxi);
  Kokkos::parallel_reduce(
      Kokkos::ThreadVectorRange(teamMember, ngnod),
      [=](const int &in, type_real &update_zxi) {
        update_zxi += dershape3D[0][in] * s_coorg(2, in);
      },
      zxi);

  Kokkos::parallel_reduce(
      Kokkos::ThreadVectorRange(teamMember, ngnod),
      [=](const int &in, type_real &update_xeta) {
        update_xeta += dershape3D[1][in] * s_coorg(0, in);
      },
      xeta);
  Kokkos::parallel_reduce(
      Kokkos::ThreadVectorRange(teamMember, ngnod),
      [=](const int &in, type_real &update_yeta) {
        update_yeta += dershape3D[1][in] * s_coorg(1, in);
      },
      yeta);
  Kokkos::parallel_reduce(
      Kokkos::ThreadVectorRange(teamMember, ngnod),
      [=](const int &in, type_real &update_zeta) {
        update_zeta += dershape3D[1][in] * s_coorg(2, in);
      },
      zeta);

  Kokkos::parallel_reduce(
      Kokkos::ThreadVectorRange(teamMember, ngnod),
      [=](const int &in, type_real &update_xgamma) {
        update_xgamma += dershape3D[2][in] * s_coorg(0, in);
      },
      xgamma);
  Kokkos::parallel_reduce(
      Kokkos::ThreadVectorRange(teamMember, ngnod),
      [=](const int &in, type_real &update_ygamma) {
        update_ygamma += dershape3D[2][in] * s_coorg(1, in);
      },
      ygamma);
  Kokkos::parallel_reduce(
      Kokkos::ThreadVectorRange(teamMember, ngnod),
      [=](const int &in, type_real &update_zgamma) {
        update_zgamma += dershape3D[2][in] * s_coorg(2, in);
      },
      zgamma);

  return std::make_tuple(xxi, yxi, zxi, xeta, yeta, zeta, xgamma, ygamma,
                         zgamma);
}

std::tuple<type_real, type_real, type_real, type_real, type_real, type_real,
           type_real, type_real, type_real>
specfem::jacobian::compute_jacobian_matrix(
    const Kokkos::View<type_real ***, Kokkos::LayoutLeft, HostSpace> &s_coorg,
    const int ngnod, const type_real xi, const type_real eta,
    const type_real gamma) {

  ASSERT(s_coorg.extent(0) == ndim, "Dimension mismatch");
  ASSERT(s_coorg.extent(1) == ngnod, "Number of nodes mismatch");

  type_real xxi = 0.0;
  type_real yxi = 0.0;
  type_real zxi = 0.0;
  type_real xeta = 0.0;
  type_real yeta = 0.0;
  type_real zeta = 0.0;
  type_real xgamma = 0.0;
  type_real ygamma = 0.0;
  type_real zgamma = 0.0;

  const auto dershape2D =
      specfem::shape_function::shape_function_derivatives(xi, gamma, ngnod);

  for (int in = 0; in < ngnod; in++) {
    xxi += dershape2D[0][in] * s_coorg(0, in);
    yxi += dershape2D[0][in] * s_coorg(1, in);
    zxi += dershape2D[0][in] * s_coorg(2, in);
    xeta += dershape2D[1][in] * s_coorg(0, in);
    yeta += dershape2D[1][in] * s_coorg(1, in);
    zeta += dershape2D[1][in] * s_coorg(2, in);
    xgamma += dershape2D[2][in] * s_coorg(0, in);
    ygamma += dershape2D[2][in] * s_coorg(1, in);
    zgamma += dershape2D[2][in] * s_coorg(2, in);
  }

  return std::make_tuple(xxi, yxi, zxi, xeta, yeta, zeta, xgamma, ygamma,
                         zgamma);
}

type_real specfem::jacobian::compute_jacobian(
    const type_real xxi, const type_real yxi, const type_real zxi,
    const type_real xeta, const type_real yeta, const type_real zeta,
    const type_real xgamma, const type_real ygamma, const type_real zgamma) {
  return xxi * (yeta * zgamma - ygamma * zeta) -
         xeta * (yxi * zgamma - ygamma * zxi) +
         xgamma * (yxi * zeta - yeta * zxi);
}

type_real specfem::jacobian::compute_jacobian(
    const specfem::kokkos::HostTeam::member_type &teamMember,
    const Kokkos::View<type_real ***, Kokkos::LayoutLeft, HostScratchSpace,
                       Kokkos::MemoryTraits<Kokkos::Unmanaged> > &s_coorg,
    const int ngnod, const type_real xi, const type_real eta,
    const type_real gamma) {
  auto [xxi, yxi, zxi, xeta, yeta, zeta, xgamma, ygamma, zgamma] =
      specfem::jacobian::compute_jacobian_matrix(teamMember, s_coorg, ngnod, xi,
                                                 eta, gamma);
  return specfem::jacobian::compute_jacobian(xxi, yxi, zxi, xeta, yeta, zeta,
                                             xgamma, ygamma, zgamma);
};

type_real specfem::jacobian::compute_jacobian(
    const specfem::kokkos::HostTeam::member_type &teamMember,
    const Kokkos::View<type_real ***, Kokkos::LayoutLeft, HostScratchSpace,
                       Kokkos::MemoryTraits<Kokkos::Unmanaged> > &s_coorg,
    const int ngnod,
    const Kokkos::View<type_real ***, Kokkos::LayoutLeft, HostSpace>
        &dershape2D) {
  auto [xxi, yxi, zxi, xeta, yeta, zeta, xgamma, ygamma, zgamma] =
      specfem::jacobian::compute_jacobian_matrix(teamMember, s_coorg, ngnod,
                                                 dershape2D);
  return specfem::jacobian::compute_jacobian(xxi, yxi, zxi, xeta, yeta, zeta,
                                             xgamma, ygamma, zgamma);
};

std::tuple<type_real, type_real, type_real, type_real>
specfem::jacobian::compute_inverted_derivatives(
    const specfem::kokkos::HostTeam::member_type &teamMember,
    const Kokkos::View<type_real ***, Kokkos::LayoutLeft, HostScratchSpace,
                       Kokkos::MemoryTraits<Kokkos::Unmanaged> > &s_coorg,
    const int ngnod, const type_real xi, const type_real eta,
    const type_real gamma) {

  auto [xxi, yxi, zxi, xeta, yeta, zeta, xgamma, ygamma, zgamma] =
      specfem::jacobian::compute_jacobian_matrix(teamMember, s_coorg, ngnod, xi,
                                                 eta, gamma);
  auto jacobian = specfem::jacobian::compute_jacobian(
      xxi, yxi, zxi, xeta, yeta, zeta, xgamma, ygamma, zgamma);

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

specfem::point::jacobian_matrix<specfem::dimension::type::dim3, true, false>
specfem::jacobian::compute_derivatives(
    const specfem::kokkos::HostTeam::member_type &teamMember,
    const Kokkos::View<type_real ***, Kokkos::LayoutLeft, HostScratchSpace,
                       Kokkos::MemoryTraits<Kokkos::Unmanaged> > &s_coorg,
    const int ngnod,
    const Kokkos::View<type_real ***, Kokkos::LayoutLeft, HostSpace>
        &dershape3D) {

  auto [xxi, yxi, zxi, xeta, yeta, zeta, xgamma, ygamma, zgamma] =
      specfem::jacobian::compute_jacobian_matrix(teamMember, s_coorg, ngnod,
                                                 dershape3D);
  auto jacobian = specfem::jacobian::compute_jacobian(
      xxi, yxi, zxi, xeta, yeta, zeta, xgamma, ygamma, zgamma);

  type_real xix = (yeta * zgamma - ygamma * zeta) / jacobian;
  type_real xiy = (xgamma * zeta - xeta * zgamma) / jacobian;
  type_real xiz = (xeta * ygamma - xgamma * yeta) / jacobian;
  type_real etax = (ygamma * zxi - yxi * zgamma) / jacobian;
  type_real etay = (xxi * zgamma - xgamma * zxi) / jacobian;
  type_real etaz = (xgamma * yxi - xxi * ygamma) / jacobian;
  type_real gammax = (yxi * zeta - yeta * zxi) / jacobian;
  type_real gammay = (xeta * zxi - xxi * zeta) / jacobian;
  type_real gammaz = (xxi * yeta - xeta * yxi) / jacobian;

  return specfem::point::jacobian_matrix<specfem::dimension::type::dim3, true,
                                         false>(
      xix, etax, gammax, xiy, etay, gammay, xiz, etaz, gammaz, jacobian);
}

std::tuple<type_real, type_real, type_real, type_real>
specfem::jacobian::compute_inverted_derivatives(
    const specfem::kokkos::HostTeam::member_type &teamMember,
    const Kokkos::View<type_real ***, Kokkos::LayoutLeft, HostScratchSpace,
                       Kokkos::MemoryTraits<Kokkos::Unmanaged> > &s_coorg,
    const int ngnod,
    const Kokkos::View<type_real ***, Kokkos::LayoutLeft, HostSpace>
        &dershape3D) {
  auto [xxi, yxi, zxi, xeta, yeta, zeta, xgamma, ygamma, zgamma] =
      specfem::jacobian::compute_jacobian_matrix(teamMember, s_coorg, ngnod,
                                                 dershape3D);
  auto jacobian = specfem::jacobian::compute_jacobian(
      xxi, yxi, zxi, xeta, yeta, zeta, xgamma, ygamma, zgamma);

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

std::tuple<type_real, type_real, type_real, type_real>
specfem::jacobian::compute_inverted_derivatives(
    const Kokkos::View<type_real ***, Kokkos::LayoutLeft, HostSpace> &s_coorg,
    const int ngnod, const type_real xi, const type_real gamma) {
  auto [xxi, yxi, zxi, xeta, yeta, zeta, xgamma, ygamma, zgamma] =
      specfem::jacobian::compute_jacobian_matrix(s_coorg, ngnod, xi, gamma);
  auto jacobian = specfem::jacobian::compute_jacobian(
      xxi, yxi, zxi, xeta, yeta, zeta, xgamma, ygamma, zgamma);

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
