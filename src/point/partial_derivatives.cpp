
#include "point/partial_derivatives.hpp"
#include "point/partial_derivatives.tpp"
#include <Kokkos_Core.hpp>

// Explicit template instantiation

template struct specfem::point::partial_derivatives2<false, false>;
template struct specfem::point::partial_derivatives2<true, false>;
template struct specfem::point::partial_derivatives2<false, true>;
template struct specfem::point::partial_derivatives2<true, true>;

// template KOKKOS_FUNCTION specfem::point::partial_derivatives2<false, false>
// specfem::point::operator+<false>(
//     const specfem::point::partial_derivatives2<false, false> &lhs,
//     const specfem::point::partial_derivatives2<false, false> &rhs);

// template KOKKOS_FUNCTION specfem::point::partial_derivatives2<true, false>
// specfem::point::operator+<true>(
//     const specfem::point::partial_derivatives2<true, false> &lhs,
//     const specfem::point::partial_derivatives2<true, false> &rhs);

// template KOKKOS_FUNCTION specfem::point::partial_derivatives2<false, false> &
// specfem::point::operator+=<false>(
//     specfem::point::partial_derivatives2<false, false> &lhs,
//     const specfem::point::partial_derivatives2<false, false> &rhs);

// template KOKKOS_FUNCTION specfem::point::partial_derivatives2<true, false> &
// specfem::point::operator+=<true>(
//     specfem::point::partial_derivatives2<true, false> &lhs,
//     const specfem::point::partial_derivatives2<true, false> &rhs);

// template KOKKOS_FUNCTION specfem::point::partial_derivatives2<false, false>
// specfem::point::operator*<false>(
//     const type_real &lhs,
//     const specfem::point::partial_derivatives2<false, false> &rhs);

// template KOKKOS_FUNCTION specfem::point::partial_derivatives2<true, false>
// specfem::point::operator*<true>(
//     const type_real &lhs,
//     const specfem::point::partial_derivatives2<true, false> &rhs);

// template KOKKOS_FUNCTION specfem::point::partial_derivatives2<false, false>
// specfem::point::operator*<false>(
//     const specfem::point::partial_derivatives2<false, false> &lhs,
//     const type_real &rhs);

// template KOKKOS_FUNCTION specfem::point::partial_derivatives2<true, false>
// specfem::point::operator*<true>(
//     const specfem::point::partial_derivatives2<true, false> &lhs,
//     const type_real &rhs);
