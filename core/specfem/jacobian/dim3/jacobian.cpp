// #include "specfem/jacobian.hpp"
// #include "macros.hpp"
// #include "specfem/shape_functions.hpp"

// namespace specfem::jacobian {
// CoordTuple3D
// compute_locations(const specfem::kokkos::HostTeam::member_type &teamMember,
//                   const HostScratchCoord &s_coorg, const int ngnod,
//                   const type_real xi, const type_real eta,
//                   const type_real gamma) {

//   ASSERT(s_coorg.extent(0) == ndim, "Dimension mismatch");
//   ASSERT(s_coorg.extent(1) == ngnod, "Number of nodes mismatch");

//   auto shape3D = specfem::shape_function::shape_function(xi, eta, gamma,
//   ngnod);

//   // FIXME:: Multi reduction is not yet implemented in kokkos
//   // This is hacky way of doing this using double vector loops
//   // Use multiple reducers once kokkos enables the feature
//   return compute_locations(teamMember, s_coorg, ngnod, shape3D);
// }

// CoordTuple3D compute_locations(const HostCoord &coorg, const int ngnod,
//                                const type_real xi, const type_real eta,
//                                const type_real gamma) {

//   ASSERT(coorg.extent(0) == ndim, "Dimension mismatch");
//   ASSERT(coorg.extent(1) == ngnod, "Number of nodes mismatch");

//   auto shape3D = specfem::shape_function::shape_function(xi, eta, gamma,
//   ngnod);

//   return compute_locations(coorg, ngnod, shape3D);
// }

// CoordTuple3D
// compute_locations(const specfem::kokkos::HostTeam::member_type &teamMember,
//                   const HostScratchCoord &s_coorg, const int ngnod,
//                   const std::vector<type_real> &shape3D) {

//   ASSERT(s_coorg.extent(0) == ndim, "Dimension mismatch");
//   ASSERT(s_coorg.extent(1) == ngnod, "Number of nodes mismatch");
//   ASSERT(shape3D.size() == ngnod, "Number of nodes mismatch");

//   type_real xcor = 0.0;
//   type_real ycor = 0.0;
//   type_real zcor = 0.0;

//   // FIXME:: Multi reduction is not yet implemented in kokkos
//   // This is hacky way of doing this using double vector loops
//   // Use multiple reducers once kokkos enables the feature

//   for (int in = 0; in < ngnod; in++) {
//     xcor += shape3D[in] * s_coorg(0, in);
//     ycor += shape3D[in] * s_coorg(1, in);
//     zcor += shape3D[in] * s_coorg(2, in);
//   }

//   return std::make_tuple(xcor, ycor, zcor);
// }

// CoordTuple3D compute_locations(const HostCoord &s_coorg, const int ngnod,
//                                const std::vector<type_real> &shape3D) {

//   ASSERT(s_coorg.extent(0) == ndim, "Dimension mismatch");
//   ASSERT(s_coorg.extent(1) == ngnod, "Number of nodes mismatch");
//   ASSERT(shape3D.size() == ngnod, "Number of nodes mismatch");

//   type_real xcor = 0.0;
//   type_real ycor = 0.0;
//   type_real zcor = 0.0;

//   for (int in = 0; in < ngnod; in++) {
//     xcor += shape3D[in] * s_coorg(0, in);
//     ycor += shape3D[in] * s_coorg(1, in);
//     zcor += shape3D[in] * s_coorg(2, in);
//   }

//   return std::make_tuple(xcor, ycor, zcor);
// }

// MatrixTuple3D compute_jacobian_matrix(
//     const specfem::kokkos::HostTeam::member_type &teamMember,
//     const HostScratchCoord &s_coorg, const int ngnod, const type_real xi,
//     const type_real eta, const type_real gamma) {

//   ASSERT(s_coorg.extent(0) == ndim, "Dimension mismatch");
//   ASSERT(s_coorg.extent(1) == ngnod, "Number of nodes mismatch");

//   type_real xxi = 0.0;
//   type_real yxi = 0.0;
//   type_real zxi = 0.0;
//   type_real xeta = 0.0;
//   type_real yeta = 0.0;
//   type_real zeta = 0.0;
//   type_real xgamma = 0.0;
//   type_real ygamma = 0.0;
//   type_real zgamma = 0.0;

//   const auto dershape3D =
//   specfem::shape_function::shape_function_derivatives(
//       xi, eta, gamma, ngnod);

//   // FIXME:: Multi reduction is not yet implemented in kokkos
//   // This is hacky way of doing this using double vector loops
//   // Use multiple reducers once kokkos enables the feature

//   Kokkos::parallel_reduce(
//       Kokkos::ThreadVectorRange(teamMember, ngnod),
//       [=](const int &in, type_real &update_xxi) {
//         update_xxi += dershape3D[0][in] * s_coorg(0, in);
//       },
//       xxi);
//   Kokkos::parallel_reduce(
//       Kokkos::ThreadVectorRange(teamMember, ngnod),
//       [=](const int &in, type_real &update_yxi) {
//         update_yxi += dershape3D[0][in] * s_coorg(1, in);
//       },
//       yxi);
//   Kokkos::parallel_reduce(
//       Kokkos::ThreadVectorRange(teamMember, ngnod),
//       [=](const int &in, type_real &update_zxi) {
//         update_zxi += dershape3D[0][in] * s_coorg(2, in);
//       },
//       zxi);

//   Kokkos::parallel_reduce(
//       Kokkos::ThreadVectorRange(teamMember, ngnod),
//       [=](const int &in, type_real &update_xeta) {
//         update_xeta += dershape3D[1][in] * s_coorg(0, in);
//       },
//       xeta);
//   Kokkos::parallel_reduce(
//       Kokkos::ThreadVectorRange(teamMember, ngnod),
//       [=](const int &in, type_real &update_yeta) {
//         update_yeta += dershape3D[1][in] * s_coorg(1, in);
//       },
//       yeta);
//   Kokkos::parallel_reduce(
//       Kokkos::ThreadVectorRange(teamMember, ngnod),
//       [=](const int &in, type_real &update_zeta) {
//         update_zeta += dershape3D[1][in] * s_coorg(2, in);
//       },
//       zeta);

//   Kokkos::parallel_reduce(
//       Kokkos::ThreadVectorRange(teamMember, ngnod),
//       [=](const int &in, type_real &update_xgamma) {
//         update_xgamma += dershape3D[2][in] * s_coorg(0, in);
//       },
//       xgamma);
//   Kokkos::parallel_reduce(
//       Kokkos::ThreadVectorRange(teamMember, ngnod),
//       [=](const int &in, type_real &update_ygamma) {
//         update_ygamma += dershape3D[2][in] * s_coorg(1, in);
//       },
//       ygamma);
//   Kokkos::parallel_reduce(
//       Kokkos::ThreadVectorRange(teamMember, ngnod),
//       [=](const int &in, type_real &update_zgamma) {
//         update_zgamma += dershape3D[2][in] * s_coorg(2, in);
//       },
//       zgamma);

//   return std::make_tuple(xxi, yxi, zxi, xeta, yeta, zeta, xgamma, ygamma,
//                          zgamma);
// }

// MatrixTuple3D compute_jacobian_matrix(
//     const specfem::kokkos::HostTeam::member_type &teamMember,
//     const HostScratchCoord &s_coorg, const int ngnod,
//     const HostCoord &dershape3D) {

//   ASSERT(s_coorg.extent(0) == ndim, "Dimension mismatch");
//   ASSERT(s_coorg.extent(1) == ngnod, "Number of nodes mismatch");
//   ASSERT(dershape3D.extent(0) == ndim, "Dimension mismatch");
//   ASSERT(dershape3D.extent(1) == ngnod, "Number of nodes mismatch");

//   type_real xxi = 0.0;
//   type_real yxi = 0.0;
//   type_real zxi = 0.0;
//   type_real xeta = 0.0;
//   type_real yeta = 0.0;
//   type_real zeta = 0.0;
//   type_real xgamma = 0.0;
//   type_real ygamma = 0.0;
//   type_real zgamma = 0.0;

//   // FIXME:: Multi reduction is not yet implemented in kokkos
//   // This is hacky way of doing this using double vector loops
//   // Use multiple reducers once kokkos enables the feature

//   Kokkos::parallel_reduce(
//       Kokkos::ThreadVectorRange(teamMember, ngnod),
//       [=](const int &in, type_real &update_xxi) {
//         update_xxi += dershape3D(0, in) * s_coorg(0, in);
//       },
//       xxi);
//   Kokkos::parallel_reduce(
//       Kokkos::ThreadVectorRange(teamMember, ngnod),
//       [=](const int &in, type_real &update_yxi) {
//         update_yxi += dershape3D(0, in) * s_coorg(1, in);
//       },
//       yxi);
//   Kokkos::parallel_reduce(
//       Kokkos::ThreadVectorRange(teamMember, ngnod),
//       [=](const int &in, type_real &update_zxi) {
//         update_zxi += dershape3D(0, in) * s_coorg(2, in);
//       },
//       zxi);

//   Kokkos::parallel_reduce(
//       Kokkos::ThreadVectorRange(teamMember, ngnod),
//       [=](const int &in, type_real &update_xeta) {
//         update_xeta += dershape3D(1, in) * s_coorg(0, in);
//       },
//       xeta);
//   Kokkos::parallel_reduce(
//       Kokkos::ThreadVectorRange(teamMember, ngnod),
//       [=](const int &in, type_real &update_yeta) {
//         update_yeta += dershape3D(1, in) * s_coorg(1, in);
//       },
//       yeta);
//   Kokkos::parallel_reduce(
//       Kokkos::ThreadVectorRange(teamMember, ngnod),
//       [=](const int &in, type_real &update_zeta) {
//         update_zeta += dershape3D(1, in) * s_coorg(2, in);
//       },
//       zeta);

//   Kokkos::parallel_reduce(
//       Kokkos::ThreadVectorRange(teamMember, ngnod),
//       [=](const int &in, type_real &update_xgamma) {
//         update_xgamma += dershape3D(2, in) * s_coorg(0, in);
//       },
//       xgamma);
//   Kokkos::parallel_reduce(
//       Kokkos::ThreadVectorRange(teamMember, ngnod),
//       [=](const int &in, type_real &update_ygamma) {
//         update_ygamma += dershape3D(2, in) * s_coorg(1, in);
//       },
//       ygamma);
//   Kokkos::parallel_reduce(
//       Kokkos::ThreadVectorRange(teamMember, ngnod),
//       [=](const int &in, type_real &update_zgamma) {
//         update_zgamma += dershape3D(2, in) * s_coorg(2, in);
//       },
//       zgamma);

//   return std::make_tuple(xxi, yxi, zxi, xeta, yeta, zeta, xgamma, ygamma,
//                          zgamma);
// }

// MatrixTuple3D compute_jacobian_matrix(const HostCoord &s_coorg, const int
// ngnod,
//                                       const type_real xi, const type_real
//                                       eta, const type_real gamma) {

//   ASSERT(s_coorg.extent(0) == ndim, "Dimension mismatch");
//   ASSERT(s_coorg.extent(1) == ngnod, "Number of nodes mismatch");

//   type_real xxi = 0.0;
//   type_real yxi = 0.0;
//   type_real zxi = 0.0;
//   type_real xeta = 0.0;
//   type_real yeta = 0.0;
//   type_real zeta = 0.0;
//   type_real xgamma = 0.0;
//   type_real ygamma = 0.0;
//   type_real zgamma = 0.0;

//   const auto dershape3D =
//       specfem::shape_function::shape_function_derivatives(xi, gamma, ngnod);

//   for (int in = 0; in < ngnod; in++) {
//     xxi += dershape3D[0][in] * s_coorg(0, in);
//     yxi += dershape3D[0][in] * s_coorg(1, in);
//     zxi += dershape3D[0][in] * s_coorg(2, in);
//     xeta += dershape3D[1][in] * s_coorg(0, in);
//     yeta += dershape3D[1][in] * s_coorg(1, in);
//     zeta += dershape3D[1][in] * s_coorg(2, in);
//     xgamma += dershape3D[2][in] * s_coorg(0, in);
//     ygamma += dershape3D[2][in] * s_coorg(1, in);
//     zgamma += dershape3D[2][in] * s_coorg(2, in);
//   }

//   return std::make_tuple(xxi, yxi, zxi, xeta, yeta, zeta, xgamma, ygamma,
//                          zgamma);
// }

// type_real compute_jacobian(const type_real xxi, const type_real yxi,
//                            const type_real zxi, const type_real xeta,
//                            const type_real yeta, const type_real zeta,
//                            const type_real xgamma, const type_real ygamma,
//                            const type_real zgamma) {
//   return xxi * (yeta * zgamma - ygamma * zeta) -
//          xeta * (yxi * zgamma - ygamma * zxi) +
//          xgamma * (yxi * zeta - yeta * zxi);
// }

// type_real
// compute_jacobian(const specfem::kokkos::HostTeam::member_type &teamMember,
//                  const HostScratchCoord &s_coorg, const int ngnod,
//                  const type_real xi, const type_real eta,
//                  const type_real gamma) {
//   auto [xxi, yxi, zxi, xeta, yeta, zeta, xgamma, ygamma, zgamma] =
//       compute_jacobian_matrix(teamMember, s_coorg, ngnod, xi, eta, gamma);
//   return compute_jacobian(xxi, yxi, zxi, xeta, yeta, zeta, xgamma, ygamma,
//                           zgamma);
// };

// type_real
// compute_jacobian(const specfem::kokkos::HostTeam::member_type &teamMember,
//                  const HostScratchCoord &s_coorg, const int ngnod,
//                  const HostCoord &dershape3D) {
//   auto [xxi, yxi, zxi, xeta, yeta, zeta, xgamma, ygamma, zgamma] =
//       compute_jacobian_matrix(teamMember, s_coorg, ngnod, dershape3D);
//   return compute_jacobian(xxi, yxi, zxi, xeta, yeta, zeta, xgamma, ygamma,
//                           zgamma);
// };

// specfem::point::jacobian_matrix<specfem::dimension::type::dim3, true, false>
// compute_derivatives(const specfem::kokkos::HostTeam::member_type &teamMember,
//                     const HostScratchCoord &s_coorg, const int ngnod,
//                     const HostCoord &dershape3D) {

//   auto [xxi, yxi, zxi, xeta, yeta, zeta, xgamma, ygamma, zgamma] =
//       compute_jacobian_matrix(teamMember, s_coorg, ngnod, dershape3D);
//   auto jacobian =
//       compute_jacobian(xxi, yxi, zxi, xeta, yeta, zeta, xgamma, ygamma,
//       zgamma);

//   type_real xix = (yeta * zgamma - ygamma * zeta) / jacobian;
//   type_real xiy = (xgamma * zeta - xeta * zgamma) / jacobian;
//   type_real xiz = (xeta * ygamma - xgamma * yeta) / jacobian;
//   type_real etax = (ygamma * zxi - yxi * zgamma) / jacobian;
//   type_real etay = (xxi * zgamma - xgamma * zxi) / jacobian;
//   type_real etaz = (xgamma * yxi - xxi * ygamma) / jacobian;
//   type_real gammax = (yxi * zeta - yeta * zxi) / jacobian;
//   type_real gammay = (xeta * zxi - xxi * zeta) / jacobian;
//   type_real gammaz = (xxi * yeta - xeta * yxi) / jacobian;

//   return specfem::point::jacobian_matrix<specfem::dimension::type::dim3,
//   true,
//                                          false>(
//       xix, etax, gammax, xiy, etay, gammay, xiz, etaz, gammaz, jacobian);
// }

// MatrixTuple3D compute_inverted_derivatives(const HostCoord &s_coorg,
//                                            const int ngnod, const type_real
//                                            xi, const type_real eta, const
//                                            type_real gamma) {
//   auto [xxi, yxi, zxi, xeta, yeta, zeta, xgamma, ygamma, zgamma] =
//       compute_jacobian_matrix(s_coorg, ngnod, xi, eta, gamma);
//   auto jacobian =
//       compute_jacobian(xxi, yxi, zxi, xeta, yeta, zeta, xgamma, ygamma,
//       zgamma);

//   type_real xix = (yeta * zgamma - ygamma * zeta) / jacobian;
//   type_real xiy = (xgamma * zeta - xeta * zgamma) / jacobian;
//   type_real xiz = (xeta * ygamma - xgamma * yeta) / jacobian;
//   type_real etax = (ygamma * zxi - yxi * zgamma) / jacobian;
//   type_real etay = (xxi * zgamma - xgamma * zxi) / jacobian;
//   type_real etaz = (xgamma * yxi - xxi * ygamma) / jacobian;
//   type_real gammax = (yxi * zeta - yeta * zxi) / jacobian;
//   type_real gammay = (xeta * zxi - xxi * zeta) / jacobian;
//   type_real gammaz = (xxi * yeta - xeta * yxi) / jacobian;

//   return std::make_tuple(xix, etax, gammax, xiy, etay, gammay, xiz, etaz,
//                          gammaz);
// }
// } // namespace specfem::jacobian
