#ifndef CONFIG_H
#define CONFIG_H

#include <Kokkos_Core.hpp>

using type_real = float;
const static int ndim{ 2 };
const static int fint{ 4 }, fdouble{ 8 }, fbool{ 4 }, fchar{ 512 };
const static bool use_best_location{ true };

KOKKOS_INLINE_FUNCTION void sub2ind(const int xz, const int ngllx, int &iz,
                                    int &ix) {
  iz = xz / ngllx;
  ix = xz - iz * ngllx;
  return;
}

#endif
