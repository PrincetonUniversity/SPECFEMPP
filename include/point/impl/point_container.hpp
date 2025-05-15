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

#define POINT_OPERATOR_DEFINITION(seq)                                         \
  KOKKOS_INLINE_FUNCTION const value_type operator[](const int i) const {      \
    return _point_data_container[i];                                           \
  }                                                                            \
  KOKKOS_INLINE_FUNCTION value_type &operator[](const int i) {                 \
    return _point_data_container[i];                                           \
  }

#define POINT_BOOLEAN_OPERATOR_DEFINITION(seq)                                 \
  template <typename U = simd>                                                 \
  typename std::enable_if_t<!U::using_simd, bool> operator==(                  \
      const data_container &other) const {                                     \
    if (nprops != other.nprops) {                                              \
      return false;                                                            \
    }                                                                          \
    for (int i = 0; i < nprops; ++i) {                                         \
      if (std::abs(_point_data_container[i] -                                  \
                   other._point_data_container[i]) >                           \
          static_cast<type_real>(1e-6) * std::abs(_point_data_container[i])) { \
        return false;                                                          \
      }                                                                        \
    }                                                                          \
    return true;                                                               \
  }

#define POINT_BOOLEAN_OPERATOR_DEFINITION_SIMD(seq)                            \
  template <bool OtherSIMD, typename U = simd>                                 \
  typename std::enable_if_t<U::using_simd, bool> operator==(                   \
      const data_container<base_type::dimension, base_type::medium_tag,        \
                           base_type::property_tag, OtherSIMD> &other) const { \
    if (nprops != other.nprops) {                                              \
      return false;                                                            \
    }                                                                          \
    for (int i = 0; i < nprops; ++i) {                                         \
      if (!Kokkos::Experimental::all_of(                                       \
              Kokkos::abs(_point_data_container[i] -                           \
                          other._point_data_container[i]) <                    \
              static_cast<type_real>(1e-6) *                                   \
                  Kokkos::abs(_point_data_container[i]))) {                    \
        return false;                                                          \
      }                                                                        \
    }                                                                          \
    return true;                                                               \
  }

#define POINT_CONSTRUCTOR(seq)                                                 \
  KOKKOS_INLINE_FUNCTION data_container() = default;                           \
  template <typename... Args,                                                  \
            typename std::enable_if_t<sizeof...(Args) == nprops, int> = 0>     \
  KOKKOS_INLINE_FUNCTION data_container(Args... args)                          \
      : _point_data_container{ args... } {}                                    \
  KOKKOS_INLINE_FUNCTION data_container(const value_type *value) {             \
    for (int i = 0; i < nprops; ++i) {                                         \
      _point_data_container[i] = value[i];                                     \
    }                                                                          \
  }                                                                            \
  KOKKOS_INLINE_FUNCTION data_container(const value_type &value) {             \
    for (int i = 0; i < nprops; ++i) {                                         \
      _point_data_container[i] = value;                                        \
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

#define POINT_PRINT_VALUE_SIMD(r, data, elem)                                  \
  BOOST_PP_TUPLE_ELEM(0, data)                                                 \
      << "\n\t\t" BOOST_PP_STRINGIZE(BOOST_PP_SEQ_ELEM(0, elem)) << " = "                                                 \
                      << _point_data_container[BOOST_PP_SEQ_ELEM(1, elem)]     \
                                              [BOOST_PP_TUPLE_ELEM(1, data)];

#define POINT_PRINT_SIMD(seq)                                                  \
  template <typename U = simd>                                                 \
  typename std::enable_if_t<U::using_simd, void> print(                        \
      std::ostringstream &message) const {                                     \
    message << "\n\t Point data: ";                                            \
    for (std::size_t lane = 0; lane < simd::size(); ++lane) {                  \
      message << "\n\t Lane " << lane << ": ";                                 \
      BOOST_PP_SEQ_FOR_EACH(POINT_PRINT_VALUE_SIMD, (message, lane), seq)      \
    }                                                                          \
    message << "\n";                                                           \
  }

#define POINT_DATA_CONTAINER_NUMBERED_SEQ(seq)                                 \
public:                                                                        \
  constexpr static int nprops = BOOST_PP_SEQ_SIZE(seq);                        \
  value_type _point_data_container[nprops] = { 0 };                            \
                                                                               \
private:                                                                       \
  POINT_PRINT(seq)                                                             \
  POINT_PRINT_SIMD(seq)                                                        \
public:                                                                        \
  POINT_CONSTRUCTOR(seq)                                                       \
  POINT_BOOLEAN_OPERATOR_DEFINITION(seq)                                       \
  POINT_BOOLEAN_OPERATOR_DEFINITION_SIMD(seq)                                  \
  POINT_VALUE_ACCESSORS(seq)                                                   \
  POINT_OPERATOR_DEFINITION(seq)                                               \
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

/**
 * @brief Compile time information associated with the properties of a
 * quadrature point in a 2D
 *
 * @tparam Dimension The dimension of the medium
 * @tparam MediumTag The type of the medium
 * @tparam PropertyTag The type of the properties
 * @tparam UseSIMD Boolean indicating whether to use SIMD intrinsics
 */
template <specfem::dimension::type Dimension,
          specfem::element::medium_tag MediumTag,
          specfem::element::property_tag PropertyTag, bool UseSIMD>
struct traits {
  using simd = typename specfem::datatype::simd<type_real, UseSIMD>;
  constexpr static auto dimension = Dimension;      ///< dimension of the medium
  constexpr static auto medium_tag = MediumTag;     ///< type of the medium
  constexpr static auto property_tag = PropertyTag; ///< type of the properties
  constexpr static bool is_point_properties = true; ///< is point properties
  using value_type = typename simd::datatype;       ///< type of the properties
};

/*
 * @brief Data container to hold properties of 2D acoustic media at a quadrature
 * point
 *
 * @tparam Dimension The dimension of the medium
 * @tparam MediumTag The type of the medium
 * @tparam PropertyTag The type of the properties
 * @tparam UseSIMD Boolean indicating whether to use SIMD intrinsics
 *
 */
template <specfem::dimension::type Dimension,
          specfem::element::medium_tag MediumTag,
          specfem::element::property_tag PropertyTag, bool UseSIMD,
          typename Enable = void>
struct data_container;
} // namespace properties

namespace kernels {
/**
 * @brief Compile time information associated with the kernels of a quadrature
 * point in a 2D
 *
 * @tparam Dimension The dimension of the medium
 * @tparam MediumTag The type of the medium
 * @tparam PropertyTag The type of the properties
 * @tparam UseSIMD Boolean indicating whether to use SIMD intrinsics
 */
template <specfem::dimension::type Dimension,
          specfem::element::medium_tag MediumTag,
          specfem::element::property_tag PropertyTag, bool UseSIMD>
struct traits {
  using simd = typename specfem::datatype::simd<type_real, UseSIMD>;
  constexpr static auto dimension = Dimension;      ///< dimension of the medium
  constexpr static auto medium_tag = MediumTag;     ///< type of the medium
  constexpr static auto property_tag = PropertyTag; ///< type of the properties
  constexpr static bool is_point_kernels = true;    ///< is point kernels
  using value_type = typename simd::datatype;       ///< type of the properties
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
