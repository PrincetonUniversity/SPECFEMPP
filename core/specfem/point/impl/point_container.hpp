#pragma once

#include "enumerations/interface.hpp"
#include "specfem/data_access.hpp"
#include "specfem_setup.hpp"
#include "utilities/interface.hpp"
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
  KOKKOS_INLINE_FUNCTION bool operator==(const data_container &other) const {  \
    if (nprops != other.nprops) {                                              \
      return false;                                                            \
    }                                                                          \
    for (int i = 0; i < nprops; ++i) {                                         \
      if (!specfem::utilities::is_close(_point_data_container[i],              \
                                        other._point_data_container[i])) {     \
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
      : _point_data_container{ static_cast<value_type>(args)... } {}           \
  KOKKOS_INLINE_FUNCTION                                                       \
  data_container(const value_type *value) {                                    \
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
 * @brief Base accessor class for point properties.
 *
 * This struct serves as a base class for accessing physical properties
 * defined at quadrature points within a spectral element. It provides
 * essential type definitions and static constants used by derived
 * property containers.
 *
 * @tparam DimensionTag The dimension of the physical domain (e.g., 2D, 3D).
 * @tparam MediumTag The type of the medium (e.g., acoustic, elastic).
 * @tparam PropertyTag The specific property category.
 * @tparam UseSIMD Flag indicating whether to use SIMD vectorization.
 *
 * @code
 * template <typename T>
 * struct MyProperties : public PropertyAccessor<dim2, acoustic, T, true> {
 *     // ...
 * };
 * @endcode
 */
template <specfem::dimension::type DimensionTag,
          specfem::element::medium_tag MediumTag,
          specfem::element::property_tag PropertyTag, bool UseSIMD>
struct PropertyAccessor : public specfem::data_access::Accessor<
                              specfem::data_access::AccessorType::point,
                              specfem::data_access::DataClassType::properties,
                              DimensionTag, UseSIMD> {

public:
  /**
   * @brief Base type alias for the point properties accessor.
   */
  using base_accessor = specfem::data_access::Accessor<
      specfem::data_access::AccessorType::point,
      specfem::data_access::DataClassType::properties, DimensionTag, UseSIMD>;

  /**
   * @brief SIMD data type used for vectorized operations.
   */
  using simd = typename base_accessor::template simd<type_real>;

  /**
   * @brief Scalar value type for property data.
   */
  using value_type = typename base_accessor::template scalar_type<type_real>;

  /**
   * @brief Compile-time constant for the medium type.
   */
  constexpr static auto medium_tag = MediumTag;

  /**
   * @brief Compile-time constant for the property type.
   */
  constexpr static auto property_tag = PropertyTag;
};

/**
 * @brief Forward declaration of the data container for properties.
 *
 * This struct is specialized to hold specific physical properties
 * based on the medium and dimension.
 *
 * @tparam DimensionTag The dimension of the medium.
 * @tparam MediumTag The type of the medium.
 * @tparam PropertyTag The type of the properties.
 * @tparam UseSIMD Boolean indicating whether to use SIMD intrinsics.
 * @tparam Enable SFINAE enabler.
 */
template <specfem::dimension::type DimensionTag,
          specfem::element::medium_tag MediumTag,
          specfem::element::property_tag PropertyTag, bool UseSIMD,
          typename Enable = void>
struct data_container;
} // namespace properties

namespace kernels {

/**
 * @brief Base accessor class for point kernels.
 *
 * This struct serves as a base class for accessing kernel data
 * (precomputed integration weights, Jacobians, etc.) at quadrature points.
 * It standardizes type definitions for SIMD and scalar operations.
 *
 * @tparam DimensionTag The dimension of the physical domain.
 * @tparam MediumTag The type of the medium.
 * @tparam PropertyTag The specific property category.
 * @tparam UseSIMD Flag indicating whether to use SIMD vectorization.
 *
 * @code
 * template <typename T>
 * struct MyKernels : public KernelsAccessor<dim2, acoustic, T, true> {
 *     // ...
 * };
 * @endcode
 */
template <specfem::dimension::type DimensionTag,
          specfem::element::medium_tag MediumTag,
          specfem::element::property_tag PropertyTag, bool UseSIMD>
struct KernelsAccessor
    : public specfem::data_access::Accessor<
          specfem::data_access::AccessorType::point,
          specfem::data_access::DataClassType::kernels, DimensionTag, UseSIMD> {

  /**
   * @brief Base type alias for the point kernels accessor.
   */
  using base_type = specfem::data_access::Accessor<
      specfem::data_access::AccessorType::point,
      specfem::data_access::DataClassType::kernels, DimensionTag, UseSIMD>;

  /**
   * @brief SIMD data type used for vectorized operations.
   */
  using simd = typename base_type::template simd<type_real>;

  /**
   * @brief Scalar value type for kernel data.
   */
  using value_type = typename base_type::template scalar_type<type_real>;

  /**
   * @brief Compile-time constant for the medium type.
   */
  constexpr static auto medium_tag = MediumTag;

  /**
   * @brief Compile-time constant for the property type.
   */
  constexpr static auto property_tag = PropertyTag;
};

/**
 * @brief Forward declaration of the data container for kernels.
 *
 * This struct is specialized to hold specific kernel data
 * based on the medium and dimension.
 *
 * @tparam DimensionTag The dimension of the medium.
 * @tparam MediumTag The type of the medium.
 * @tparam PropertyTag The type of the properties.
 * @tparam UseSIMD Boolean indicating whether to use SIMD intrinsics.
 * @tparam Enable SFINAE enabler.
 */
template <specfem::dimension::type DimensionTag,
          specfem::element::medium_tag MediumTag,
          specfem::element::property_tag PropertyTag, bool UseSIMD,
          typename Enable = void>
struct data_container;
} // namespace kernels

} // namespace impl

} // namespace point
} // namespace specfem
