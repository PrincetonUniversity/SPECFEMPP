#include "point/coordinates.hpp"
#include <Kokkos_Core.hpp>
#include <cmath>

type_real specfem::point::distance(const specfem::point::gcoord2 &p1,
                                   const specfem::point::gcoord2 &p2) {
  return Kokkos::sqrt((p1.x - p2.x) * (p1.x - p2.x) +
                      (p1.z - p2.z) * (p1.z - p2.z));
}
