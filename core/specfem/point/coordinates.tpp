#pragma once

#include "coordinates.hpp"
#include <Kokkos_Core.hpp>

template <>
KOKKOS_FUNCTION type_real specfem::point::distance(
    const specfem::point::global_coordinates<specfem::dimension::type::dim2>
        &p1,
    const specfem::point::global_coordinates<specfem::dimension::type::dim2>
        &p2) {
  return Kokkos::sqrt((p1.x - p2.x) * (p1.x - p2.x) +
                      (p1.z - p2.z) * (p1.z - p2.z));
}

template <>
KOKKOS_FUNCTION type_real specfem::point::distance(
    const specfem::point::global_coordinates<specfem::dimension::type::dim3>
        &p1,
    const specfem::point::global_coordinates<specfem::dimension::type::dim3>
        &p2) {
  return Kokkos::sqrt((p1.x - p2.x) * (p1.x - p2.x) +
                      (p1.y - p2.y) * (p1.y - p2.y) +
                      (p1.z - p2.z) * (p1.z - p2.z));
}

template <>
std::ostream &operator<<(
    std::ostream &s,
    const specfem::point::global_coordinates<specfem::dimension::type::dim2>
        &coord) {
  return s << "(" << coord.x << ", " << coord.z << ")";
}

template <>
std::ostream &operator<<(
    std::ostream &s,
    const specfem::point::global_coordinates<specfem::dimension::type::dim3>
        &coord) {
  return s << "(" << coord.x << ", " << coord.y << ", " << coord.z << ")";
}
