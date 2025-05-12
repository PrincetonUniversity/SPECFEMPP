#pragma once

#include "datatypes/simd.hpp"
#include "enumerations/dimension.hpp"
#include "enumerations/medium.hpp"
#include <Kokkos_SIMD.hpp>
#include <boost/preprocessor.hpp>
#include <iostream>
#include <sstream>

#define POINT_VALUE_ACCESSOR(r, data, elem)                                    \
  KOKKOS_INLINE_FUNCTION                                                       \
  const value_type BOOST_PP_SEQ_ELEM(0, elem)() const {                        \
    return _point_data_container[BOOST_PP_SEQ_ELEM(1, elem)];                  \
  }                                                                            \
  KOKKOS_INLINE_FUNCTION                                                       \
  value_type &BOOST_PP_SEQ_ELEM(0, elem)() {                                   \
    return _point_data_container[BOOST_PP_SEQ_ELEM(1, elem)];                  \
  }

#define POINT_VALUE_ACCESSORS(seq)                                             \
  BOOST_PP_SEQ_FOR_EACH(POINT_VALUE_ACCESSOR, _, seq)

#define POINT_VALUE_DEFINITION(seq)                                            \
  value_type _point_data_container[BOOST_PP_SEQ_SIZE(seq)] = { 0 };

#define POINT_OPERATOR_DEFINITION(seq)                                         \
  KOKKOS_INLINE_FUNCTION const value_type operator[](const int i) const {      \
    return _point_data_container[i];                                           \
  }                                                                            \
  KOKKOS_INLINE_FUNCTION value_type &operator[](const int i) {                 \
    return _point_data_container[i];                                           \
  }

#define POINT_BOOLEAN_OPERATOR_DEFINITION(seq)                                 \
  KOKKOS_INLINE_FUNCTION bool equal_operator_dispatch(                         \
      const std::integral_constant<bool, false> &,                             \
      const data_container &other) const {                                     \
    for (int i = 0; i < BOOST_PP_SEQ_SIZE(seq); ++i) {                         \
      if (std::abs(_point_data_container[i] -                                  \
                   other._point_data_container[i]) >                           \
          1e-6 * std::abs(_point_data_container[i])) {                         \
        return false;                                                          \
      }                                                                        \
    }                                                                          \
    return true;                                                               \
  }

#define POINT_BOOLEAN_OPERATOR_DEFINITION_SIMD(seq)                            \
  KOKKOS_INLINE_FUNCTION bool equal_operator_dispatch(                         \
      const std::integral_constant<bool, true> &, const data_container &other) \
      const {                                                                  \
    for (int i = 0; i < BOOST_PP_SEQ_SIZE(seq); ++i) {                         \
      if (!Kokkos::Experimental::all_of(                                       \
              Kokkos::abs(_point_data_container[i] -                           \
                          other._point_data_container[i]) <                    \
              1e-6 * Kokkos::abs(_point_data_container[i]))) {                 \
        return false;                                                          \
      }                                                                        \
    }                                                                          \
    return true;                                                               \
  }

#define POINT_CONSTRUCTOR(seq)                                                 \
  KOKKOS_INLINE_FUNCTION data_container() = default;                           \
  template <typename... Args,                                                  \
            typename std::enable_if_t<                                         \
                sizeof...(Args) == BOOST_PP_SEQ_SIZE(seq), int> = 0>           \
  KOKKOS_INLINE_FUNCTION data_container(Args... args)                          \
      : _point_data_container{ args... } {}                                    \
  KOKKOS_INLINE_FUNCTION data_container(const value_type *value) {             \
    for (int i = 0; i < BOOST_PP_SEQ_SIZE(seq); ++i) {                         \
      _point_data_container[i] = value[i];                                     \
    }                                                                          \
  }

#define POINT_PRINT_VALUE(r, message, elem)                                    \
  message << "\n\t\t" BOOST_PP_STRINGIZE(BOOST_PP_SEQ_ELEM(0, elem)) << " = " << _point_data_container[BOOST_PP_SEQ_ELEM(1, elem)];

#define POINT_PRINT(seq)                                                       \
  template <typename U = simd>                                                 \
  typename std::enable_if_t<!U::using_simd, void> print(                       \
      std::ostringstream &message) const {                                     \
    message << "\n\t Point data: ";                                            \
    BOOST_PP_SEQ_FOR_EACH(POINT_PRINT_VALUE, message, seq)                     \
    message << "\n";                                                           \
  }

// #define POINT_PRINT_VALUE_SIMD(r, data, elem)                                  \
//   BOOST_PP_TUPLE_ELEM(0, data)                                                 \
//       << "\n\t\t" BOOST_PP_STRINGIZE(BOOST_PP_SEQ_ELEM(0, elem)) << " = "                                                 \
//                       << _point_data_container[BOOST_PP_SEQ_ELEM(1, elem)]     \
//                                               [BOOST_PP_TUPLE_ELEM(1, data)];

// #define POINT_PRINT_SIMD(seq)                                                  \
//   template <typename U = simd>                                                 \
//   typename std::enable_if_t<U::using_simd, void> print(                        \
//       std::ostringstream &message) const {                                     \
//     message << "\n\t Point data: ";                                            \
//     for (std::size_t lane = 0; lane < simd::size(); ++lane) {                  \
//       message << "\n\t Lane " << lane << ": ";                                 \
//       BOOST_PP_SEQ_FOR_EACH(POINT_PRINT_VALUE_SIMD, (message, lane), seq)      \
//     }                                                                          \
//     message << "\n";                                                           \
//   }

#define POINT_DATA_CONTAINER_NUMBERED_SEQ(seq)                                 \
public:                                                                        \
  POINT_VALUE_DEFINITION(seq)                                                  \
private:                                                                       \
  POINT_BOOLEAN_OPERATOR_DEFINITION(seq)                                       \
  POINT_BOOLEAN_OPERATOR_DEFINITION_SIMD(seq)                                  \
  POINT_PRINT(seq)                                                             \
public:                                                                        \
  POINT_CONSTRUCTOR(seq)                                                       \
  POINT_VALUE_ACCESSORS(seq)                                                   \
  POINT_OPERATOR_DEFINITION(seq)                                               \
  KOKKOS_INLINE_FUNCTION bool operator==(const data_container &other) const {  \
    return equal_operator_dispatch(                                            \
        std::integral_constant<bool, simd::using_simd>(), other);              \
  }                                                                            \
  KOKKOS_INLINE_FUNCTION bool operator!=(const data_container &other) const {  \
    return !(*this == other);                                                  \
  }                                                                            \
  std::string print() const {                                                  \
    std::ostringstream message;                                                \
    print(message);                                                            \
    return message.str();                                                      \
  }

#define POINT_CREATE_SEQUENCE(r, data, i, elem) ((elem)(i))

#define POINT_CREATE_NUMBERED_SEQ(seq)                                         \
  (BOOST_PP_SEQ_FOR_EACH_I(POINT_CREATE_SEQUENCE, _, seq))

#define POINT_DATA_CONTAINER_SEQ(seq)                                          \
  BOOST_PP_EXPAND(                                                             \
      POINT_DATA_CONTAINER_NUMBERED_SEQ POINT_CREATE_NUMBERED_SEQ(seq))

#define POINT_ARGS(...) BOOST_PP_VARIADIC_TO_SEQ(__VA_ARGS__)

#define POINT_CONTAINER(...) POINT_DATA_CONTAINER_SEQ(POINT_ARGS(__VA_ARGS__))

namespace specfem {
namespace point {

namespace impl {
namespace properties {

template <specfem::dimension::type Dimension,
          specfem::element::medium_tag MediumTag,
          specfem::element::property_tag PropertyTag, bool UseSIMD>
struct point_traits {
  using simd = typename specfem::datatype::simd<type_real, UseSIMD>;
  constexpr static auto dimension = Dimension;
  constexpr static auto medium_tag = MediumTag;
  constexpr static auto property_tag = PropertyTag;
  constexpr static bool is_point_properties = true;
  using value_type = typename simd::datatype;
};

template <specfem::dimension::type Dimension,
          specfem::element::medium_tag MediumTag,
          specfem::element::property_tag PropertyTag, bool UseSIMD,
          typename Enable = void>
struct data_container;
} // namespace properties

namespace kernels {
template <specfem::dimension::type Dimension,
          specfem::element::medium_tag MediumTag,
          specfem::element::property_tag PropertyTag, bool UseSIMD>
struct point_traits {
  using simd = typename specfem::datatype::simd<type_real, UseSIMD>;
  constexpr static auto dimension = Dimension;
  constexpr static auto medium_tag = MediumTag;
  constexpr static auto property_tag = PropertyTag;
  constexpr static bool is_point_kernels = true;
  using value_type = typename simd::datatype;
};

template <specfem::dimension::type Dimension,
          specfem::element::medium_tag MediumTag,
          specfem::element::property_tag PropertyTag, bool UseSIMD,
          typename Enable = void>
struct data_container;
} // namespace kernels

} // namespace impl

} // namespace point
} // namespace specfem
