#ifndef SPECFEM_SETUP_H
#define SPECFEM_SETUP_H

#include <Kokkos_Core.hpp>

using type_real = float;
const static int ndim{ 2 };
const static int fint{ 4 }, fdouble{ 8 }, fbool{ 4 }, fchar{ 512 };
const static bool use_best_location{ true };

#if defined(KOKKOS_ENABLE_CUDA)
constexpr int NTHREADS = 32;
constexpr int NLANES = 1;
#elif defined(KOKKOS_ENABLE_HIP)
constexpr int NTHREADS = 32;
constexpr int NLANES = 1;
#elif defined(KOKKOS_ENABLE_SERIAL)
constexpr Kokkos::AUTO_t NTHREADS = Kokkos::AUTO;
constexpr int NLANES = 1;
#elif defined(KOKKOS_ENABLE_OPENMP)
constexpr Kokkos::AUTO_t NTHREADS = Kokkos::AUTO;
constexpr int NLANES = 1;
#endif

KOKKOS_INLINE_FUNCTION void sub2ind(const int xz, const int ngllx, int &iz,
                                    int &ix) {
  iz = xz / ngllx;
  ix = xz - iz * ngllx;
  return;
}

#endif
