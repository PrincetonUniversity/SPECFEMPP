#pragma once

#include <Kokkos_Core.hpp>

namespace specfem {
namespace point {

struct assembly_index {
  int iglob;

  KOKKOS_FUNCTION
  assembly_index() = default;

  KOKKOS_FUNCTION
  assembly_index(const int &iglob) : iglob(iglob) {}
};

struct simd_assembly_index {
  int number_points;
  int iglob;
  KOKKOS_FUNCTION
  bool mask(const std::size_t &lane) const { return int(lane) < number_points; }

  KOKKOS_FUNCTION
  simd_assembly_index() = default;

  KOKKOS_FUNCTION
  simd_assembly_index(const int &iglob, const int &number_points)
      : number_points(number_points), iglob(iglob) {}
};
} // namespace point
} // namespace specfem
