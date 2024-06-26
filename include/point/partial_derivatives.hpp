#ifndef _POINT_PARTIAL_DERIVATIVES_HPP
#define _POINT_PARTIAL_DERIVATIVES_HPP

#include "datatypes/point_view.hpp"
#include "edge/interface.hpp"
#include "enumerations/specfem_enums.hpp"
#include "macros.hpp"
#include "specfem_setup.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace point {

template <bool StoreJacobian = false> struct partial_derivatives2;

template <> struct partial_derivatives2<false> {
  type_real xix;
  type_real gammax;
  type_real xiz;
  type_real gammaz;

  KOKKOS_FUNCTION
  partial_derivatives2() = default;

  KOKKOS_FUNCTION
  partial_derivatives2(const type_real &xix, const type_real &gammax,
                       const type_real &xiz, const type_real &gammaz)
      : xix(xix), gammax(gammax), xiz(xiz), gammaz(gammaz) {}

  KOKKOS_FUNCTION
  partial_derivatives2(const type_real constant)
      : xix(constant), gammax(constant), xiz(constant), gammaz(constant) {}

  KOKKOS_FUNCTION
  void init() {
    this->xix = 0.0;
    this->gammax = 0.0;
    this->xiz = 0.0;
    this->gammaz = 0.0;
    return;
  }

  KOKKOS_FUNCTION
  partial_derivatives2(const partial_derivatives2<false> &rhs) = default;
};

template <>
struct partial_derivatives2<true> : public partial_derivatives2<false> {

  type_real jacobian;

  KOKKOS_FUNCTION
  partial_derivatives2() = default;

  KOKKOS_FUNCTION
  partial_derivatives2(const type_real &xix, const type_real &gammax,
                       const type_real &xiz, const type_real &gammaz,
                       const type_real &jacobian)
      : partial_derivatives2<false>(xix, gammax, xiz, gammaz),
        jacobian(jacobian) {}

  KOKKOS_FUNCTION
  partial_derivatives2(const type_real constant)
      : partial_derivatives2<false>(constant), jacobian(constant) {}

  KOKKOS_FUNCTION
  void init() {
    this->xix = 0.0;
    this->gammax = 0.0;
    this->xiz = 0.0;
    this->gammaz = 0.0;
    this->jacobian = 0.0;
    return;
  }

  KOKKOS_FUNCTION
  partial_derivatives2(const partial_derivatives2<true> &rhs) = default;

  KOKKOS_FUNCTION specfem::datatype::ScalarPointViewType<type_real, 2>
  compute_normal(const specfem::enums::edge::type &type) const;

  KOKKOS_FUNCTION specfem::datatype::ScalarPointViewType<type_real, 2>
  compute_normal(const specfem::edge::interface &interface) const;

private:
  template <specfem::enums::edge::type type>
  KOKKOS_INLINE_FUNCTION specfem::datatype::ScalarPointViewType<type_real, 2>
  impl_compute_normal() const {
    ASSERT(false, "Invalid boundary type");
    return {};
  };
};

// struct partial_derivatives2 {
//   type_real xix;
//   type_real gammax;
//   type_real xiz;
//   type_real gammaz;
//   type_real jacobian;

//   KOKKOS_FUNCTION
//   partial_derivatives2() = default;

//   KOKKOS_FUNCTION
//   partial_derivatives2(const type_real &xix, const type_real &gammax,
//                        const type_real &xiz, const type_real &gammaz)
//       : xix(xix), gammax(gammax), xiz(xiz), gammaz(gammaz) {}

//   KOKKOS_FUNCTION
//   partial_derivatives2(const type_real &xix, const type_real &gammax,
//                        const type_real &xiz, const type_real &gammaz,
//                        const type_real &jacobian)
//       : xix(xix), gammax(gammax), xiz(xiz), gammaz(gammaz),
//       jacobian(jacobian) {
//   }

//   KOKKOS_FUNCTION
//   partial_derivatives2(const type_real constant)
//       : xix(constant), gammax(constant), xiz(constant), gammaz(constant) {}

//   KOKKOS_FUNCTION
//   void init() {
//     this->xix = 0.0;
//     this->gammax = 0.0;
//     this->xiz = 0.0;
//     this->gammaz = 0.0;
//     return;
//   }

//   KOKKOS_FUNCTION
//   partial_derivatives2(const partial_derivatives2 &rhs) = default;

//   template <specfem::enums::edge::type type>
//   KOKKOS_INLINE_FUNCTION specfem::kokkos::array_type<type_real, 2>
//   compute_normal() const {
//     ASSERT(false, "Invalid boundary type");
//     return specfem::kokkos::array_type<type_real, 2>();
//   };

//   KOKKOS_FUNCTION specfem::kokkos::array_type<type_real, 2>
//   compute_normal(const specfem::enums::edge::type &type) const;

//   KOKKOS_FUNCTION specfem::kokkos::array_type<type_real, 2>
//   compute_normal(const specfem::edge::interface &interface) const;
// };

// operator+
KOKKOS_FUNCTION specfem::point::partial_derivatives2<false>
operator+(const partial_derivatives2<false> &lhs,
          const partial_derivatives2<false> &rhs);
// operator+=
KOKKOS_FUNCTION
specfem::point::partial_derivatives2<false> &
operator+=(partial_derivatives2<false> &lhs,
           const partial_derivatives2<false> &rhs);

// operator*
KOKKOS_FUNCTION
specfem::point::partial_derivatives2<false>
operator*(const type_real &lhs, const partial_derivatives2<false> &rhs);

// operator*
KOKKOS_FUNCTION
specfem::point::partial_derivatives2<false>
operator*(const partial_derivatives2<false> &lhs, const type_real &rhs);
} // namespace point
} // namespace specfem

#endif
