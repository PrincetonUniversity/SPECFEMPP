#pragma once

#include "datatypes/simd.hpp"
#include "specfem_setup.hpp"

namespace specfem {
namespace utilities {

// struct return_holder {
//   type_real rho, mu, kappa, qmu, qkappa, lambdaplus2mu;
// };

// std::tuple<type_real, type_real, int, int>
// locate(const specfem::kokkos::HostView2d<type_real> coord,
//        const specfem::kokkos::HostMirror3d<int> ibool,
//        const specfem::kokkos::HostMirror1d<type_real> xigll,
//        const specfem::kokkos::HostMirror1d<type_real> zigll, const int nproc,
//        const type_real x_source, const type_real z_source,
//        const specfem::kokkos::HostView2d<type_real> coorg,
//        const specfem::kokkos::HostView2d<int> knods, const int npgeo,
//        const specfem::MPI::MPI *mpi);

// void check_locations(const type_real x, const type_real z, const type_real
// xmin,
//                      const type_real xmax, const type_real zmin,
//                      const type_real zmax, const specfem::MPI::MPI *mpi);

// int compute_nglob(const specfem::kokkos::HostMirror3d<int> ibool);

/**
 * @brief Check if two values are close within a tolerance.
 *
 * This function checks whether two values are considered "close" to each other
 * within a specified tolerance. The comparison uses a relative error
 * calculation that accounts for the magnitude of the values.
 *
 * @tparam T Type of values to compare. Must support arithmetic operations.
 * @param a First value to compare.
 * @param b Second value to compare.
 * @param tol Relative tolerance for comparison (default: 1e-6).
 * @return bool True if values are considered close according to the formula:
 *              |a - b| <= tol * max(|a|, |b|)
 *              For simd types, returns true only if all lanes meet the
 * condition.
 */
template <typename T>
KOKKOS_INLINE_FUNCTION bool is_close(const T &a, const T &b,
                                     const T &tol = static_cast<T>(1e-6)) {
  return specfem::datatype::all_of(
      Kokkos::abs(a - b) <= tol * Kokkos::max(Kokkos::abs(a), Kokkos::abs(b)));
}

} // namespace utilities
} // namespace specfem
