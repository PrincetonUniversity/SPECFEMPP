#pragma once

#include "global_coordinates.hpp"
#include <Kokkos_Core.hpp>

template <>
std::ostream &operator<<(
    std::ostream &s,
    const specfem::point::global_coordinates<specfem::element::dimension_tag::dim2>
        &coord) {
  return s << "(" << coord.x << ", " << coord.z << ")";
}

template <>
std::ostream &operator<<(
    std::ostream &s,
    const specfem::point::global_coordinates<specfem::element::dimension_tag::dim3>
        &coord) {
  return s << "(" << coord.x << ", " << coord.y << ", " << coord.z << ")";
}
