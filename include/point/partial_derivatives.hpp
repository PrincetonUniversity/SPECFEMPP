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

template <bool UseSIMD, bool StoreJacobian> struct partial_derivatives2;

template <bool UseSIMD> struct partial_derivatives2<UseSIMD, false> {

  using simd = specfem::datatype::simd<type_real, UseSIMD>;
  using value_type = typename simd::datatype;
  constexpr static bool store_jacobian = false;

  value_type xix;
  value_type gammax;
  value_type xiz;
  value_type gammaz;

  KOKKOS_FUNCTION
  partial_derivatives2() = default;

  KOKKOS_FUNCTION
  partial_derivatives2(const value_type &xix, const value_type &gammax,
                       const value_type &xiz, const value_type &gammaz)
      : xix(xix), gammax(gammax), xiz(xiz), gammaz(gammaz) {}

  KOKKOS_FUNCTION
  partial_derivatives2(const value_type constant)
      : xix(constant), gammax(constant), xiz(constant), gammaz(constant) {}

  KOKKOS_FUNCTION
  void init() {
    this->xix = 0.0;
    this->gammax = 0.0;
    this->xiz = 0.0;
    this->gammaz = 0.0;
    return;
  }
};

template <bool UseSIMD>
struct partial_derivatives2<UseSIMD, true>
    : public partial_derivatives2<UseSIMD, false> {

  using simd = specfem::datatype::simd<type_real, UseSIMD>;
  using value_type = typename simd::datatype;
  constexpr static bool store_jacobian = true;

  value_type jacobian;

  KOKKOS_FUNCTION
  partial_derivatives2() = default;

  KOKKOS_FUNCTION
  partial_derivatives2(const value_type &xix, const value_type &gammax,
                       const value_type &xiz, const value_type &gammaz,
                       const value_type &jacobian)
      : partial_derivatives2<UseSIMD, false>(xix, gammax, xiz, gammaz),
        jacobian(jacobian) {}

  KOKKOS_FUNCTION
  partial_derivatives2(const value_type constant)
      : partial_derivatives2<UseSIMD, false>(constant), jacobian(constant) {}

  KOKKOS_FUNCTION
  void init() {
    this->xix = 0.0;
    this->gammax = 0.0;
    this->xiz = 0.0;
    this->gammaz = 0.0;
    this->jacobian = 0.0;
    return;
  }

  KOKKOS_FUNCTION specfem::datatype::ScalarPointViewType<type_real, 2, UseSIMD>
  compute_normal(const specfem::enums::edge::type &type) const;

  KOKKOS_FUNCTION specfem::datatype::ScalarPointViewType<type_real, 2, UseSIMD>
  compute_normal(const specfem::edge::interface &interface) const;

private:
  KOKKOS_INLINE_FUNCTION
  specfem::datatype::ScalarPointViewType<type_real, 2, UseSIMD>
  impl_compute_normal_bottom() const {
    return { static_cast<value_type>(static_cast<type_real>(-1.0) *
                                     this->gammax * this->jacobian),
             static_cast<value_type>(static_cast<type_real>(-1.0) *
                                     this->gammaz * this->jacobian) };
  };

  KOKKOS_INLINE_FUNCTION
  specfem::datatype::ScalarPointViewType<type_real, 2, UseSIMD>
  impl_compute_normal_top() const {
    return { static_cast<value_type>(this->gammax * this->jacobian),
             static_cast<value_type>(this->gammaz * this->jacobian) };
  };

  KOKKOS_INLINE_FUNCTION
  specfem::datatype::ScalarPointViewType<type_real, 2, UseSIMD>
  impl_compute_normal_left() const {
    return { static_cast<value_type>(static_cast<type_real>(-1.0) * this->xix *
                                     this->jacobian),
             static_cast<value_type>(static_cast<type_real>(-1.0) * this->xiz *
                                     this->jacobian) };
  };

  KOKKOS_INLINE_FUNCTION
  specfem::datatype::ScalarPointViewType<type_real, 2, UseSIMD>
  impl_compute_normal_right() const {
    return { static_cast<value_type>(this->xix * this->jacobian),
             static_cast<value_type>(this->xiz * this->jacobian) };
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
template <bool UseSIMD>
KOKKOS_FUNCTION specfem::point::partial_derivatives2<UseSIMD, false>
operator+(const partial_derivatives2<UseSIMD, false> &lhs,
          const partial_derivatives2<UseSIMD, false> &rhs) {
  return specfem::point::partial_derivatives2<UseSIMD, false>(
      lhs.xix + rhs.xix, lhs.gammax + rhs.gammax, lhs.xiz + rhs.xiz,
      lhs.gammaz + rhs.gammaz);
}
// operator+=
template <bool UseSIMD>
KOKKOS_FUNCTION specfem::point::partial_derivatives2<UseSIMD, false>
operator+=(partial_derivatives2<UseSIMD, false> &lhs,
           const partial_derivatives2<UseSIMD, false> &rhs) {
  lhs.xix += rhs.xix;
  lhs.gammax += rhs.gammax;
  lhs.xiz += rhs.xiz;
  lhs.gammaz += rhs.gammaz;
  return lhs;
}

// operator*
template <bool UseSIMD>
KOKKOS_FUNCTION specfem::point::partial_derivatives2<UseSIMD, false>
operator*(const type_real &lhs,
          const partial_derivatives2<UseSIMD, false> &rhs) {
  return specfem::point::partial_derivatives2<UseSIMD, false>(
      lhs * rhs.xix, lhs * rhs.gammax, lhs * rhs.xiz, lhs * rhs.gammaz);
}

// operator*
template <bool UseSIMD>
KOKKOS_FUNCTION specfem::point::partial_derivatives2<UseSIMD, false>
operator*(const partial_derivatives2<UseSIMD, false> &lhs,
          const type_real &rhs) {
  return specfem::point::partial_derivatives2<UseSIMD, false>(
      lhs.xix * rhs, lhs.gammax * rhs, lhs.xiz * rhs, lhs.gammaz * rhs);
}
} // namespace point
} // namespace specfem

#endif
