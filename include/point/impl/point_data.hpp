#pragma once

#include "datatypes/simd.hpp"
#include "enumerations/medium.hpp"
#include <Kokkos_Core.hpp>
#include <string>
#include <type_traits>

#define DEFINE_POINT_VALUE(prop, index_value)                                  \
  KOKKOS_INLINE_FUNCTION value_type &prop() {                                  \
    return base_type::data[index_value];                                       \
  }                                                                            \
  KOKKOS_INLINE_FUNCTION const value_type prop() const {                       \
    return base_type::data[index_value];                                       \
  }

namespace specfem {
namespace point {

namespace impl {
template <int N, bool UseSIMD> struct point_data {
  using simd = specfem::datatype::simd<type_real, UseSIMD>; ///< SIMD type
  using value_type = typename simd::datatype;
  constexpr static auto nprops = N;

  value_type data[N];

  KOKKOS_FUNCTION
  point_data() {};

  /**
   * @brief array constructor
   *
   */
  KOKKOS_FUNCTION
  point_data(const value_type *value) {
    for (int i = 0; i < N; ++i) {
      data[i] = value[i];
    }
  }

  /**
   * @brief value constructor
   *
   */
  template <typename... Args,
            typename std::enable_if_t<sizeof...(Args) == N, int> = 0>
  KOKKOS_FUNCTION point_data(Args... args) : data{ args... } {}

  /**
   * @brief Equality operator
   *
   */
  KOKKOS_FUNCTION
  bool operator==(const point_data<N, UseSIMD> &rhs) const {
    for (int i = 0; i < N; ++i) {
      if (Kokkos::abs(data[i] - rhs.data[i]) > 1e-6 * Kokkos::abs(data[i])) {
        return false;
      }
    }
    return true;
  }

  /**
   * @brief Inequality operator
   *
   */
  KOKKOS_FUNCTION
  bool operator!=(const point_data<N, UseSIMD> &rhs) const {
    return !(*this == rhs);
  }

  /**
   * @brief Print the data
   *
   */
  std::string print() const {
    return print(std::integral_constant<bool, UseSIMD>());
  }

private:
  std::string print(const std::integral_constant<bool, true> &) const {
    std::ostringstream message;
    message << "Data cannot be printed when simd is enabled";
    return message.str();
  }

  std::string print(const std::integral_constant<bool, false> &) const {
    std::ostringstream message;
    message << "Data: ";
    for (int i = 0; i < N; ++i) {
      message << data[i] << " ";
    }
    return message.str();
  }
};
} // namespace impl

} // namespace point
} // namespace specfem
