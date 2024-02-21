
#include "point/partial_derivatives.hpp"
#include "point/partial_derivatives.tpp"
#include <Kokkos_Core.hpp>

// operator+
KOKKOS_FUNCTION
specfem::point::partial_derivatives2
specfem::point::operator+(const specfem::point::partial_derivatives2 &lhs,
                          const specfem::point::partial_derivatives2 &rhs) {
  return specfem::point::partial_derivatives2(
      lhs.xix + rhs.xix, lhs.gammax + rhs.gammax, lhs.xiz + rhs.xiz,
      lhs.gammaz + rhs.gammaz);
}

// operator+=
KOKKOS_FUNCTION
specfem::point::partial_derivatives2 &
specfem::point::operator+=(specfem::point::partial_derivatives2 &lhs,
                           const specfem::point::partial_derivatives2 &rhs) {
  lhs.xix += rhs.xix;
  lhs.gammax += rhs.gammax;
  lhs.xiz += rhs.xiz;
  lhs.gammaz += rhs.gammaz;
  return lhs;
}

// operator*
KOKKOS_FUNCTION
specfem::point::partial_derivatives2
specfem::point::operator*(const type_real &lhs,
                          const specfem::point::partial_derivatives2 &rhs) {
  return specfem::point::partial_derivatives2(lhs * rhs.xix, lhs * rhs.gammax,
                                              lhs * rhs.xiz, lhs * rhs.gammaz);
}

// operator*
KOKKOS_FUNCTION
specfem::point::partial_derivatives2
specfem::point::operator*(const specfem::point::partial_derivatives2 &lhs,
                          const type_real &rhs) {
  return specfem::point::partial_derivatives2(lhs.xix * rhs, lhs.gammax * rhs,
                                              lhs.xiz * rhs, lhs.gammaz * rhs);
}

KOKKOS_FUNCTION specfem::kokkos::array_type<type_real, 2>
specfem::point::partial_derivatives2::compute_normal(
    const specfem::enums::edge::type &type) const {
  switch (type) {
  case specfem::enums::edge::type::BOTTOM:
    return this->compute_normal<specfem::enums::edge::type::BOTTOM>();
  case specfem::enums::edge::type::TOP:
    return this->compute_normal<specfem::enums::edge::type::TOP>();
  case specfem::enums::edge::type::LEFT:
    return this->compute_normal<specfem::enums::edge::type::LEFT>();
  case specfem::enums::edge::type::RIGHT:
    return this->compute_normal<specfem::enums::edge::type::RIGHT>();
  default:
    ASSERT(false, "Invalid boundary type");
    return specfem::kokkos::array_type<type_real, 2>();
  }
}

// compute_normal
KOKKOS_FUNCTION specfem::kokkos::array_type<type_real, 2>
specfem::point::partial_derivatives2::compute_normal(
    const specfem::edge::interface &interface) const {
  return specfem::point::partial_derivatives2::compute_normal(interface.type);
}
