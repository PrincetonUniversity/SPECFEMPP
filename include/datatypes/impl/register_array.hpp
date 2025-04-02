#pragma once

#include "../simd.hpp"
#include <Kokkos_Core.hpp>
#include <mdspan/mdspan.hpp>

namespace specfem {
namespace datatype {
namespace impl {

template <typename Extents> constexpr size_t compute_size() {

  size_t size = 1;
  for (size_t i = 0; i < Extents::rank(); ++i) {
    size *= Extents::static_extent(i);
  }
  return size;
}

template <typename T, typename Extents, typename Layout> class RegisterArray {

private:
  constexpr static std::size_t rank = Extents::rank();
  constexpr static std::size_t size = impl::compute_size<Extents>();
  using mapping = typename Layout::template mapping<Extents>;

public:
  using value_type = T;
  RegisterArray(const value_type value) {

    for (std::size_t i = 0; i < size; ++i) {
      m_value[i] = value;
    }
  }

  template <typename... Args,
            typename std::enable_if<sizeof...(Args) == size, bool>::type = true>
  RegisterArray(const Args &...args) : m_value{ args... } {}

  RegisterArray(const RegisterArray &other) {

    for (std::size_t i = 0; i < size; ++i) {
      m_value[i] = other.m_value[i];
    }
  }

  RegisterArray() {
    for (std::size_t i = 0; i < size; ++i) {
      m_value[i] = 0.0;
    }
  }

  RegisterArray(const T *values) {
    for (std::size_t i = 0; i < size; ++i) {
      m_value[i] = values[i];
    }
  }

  template <typename... IndexType>
  value_type &operator()(const IndexType &...i) {
    return m_value[mapping()(i...)];
  }

  template <typename... IndexType>
  const value_type &operator()(const IndexType &...i) const {
    return m_value[mapping()(i...)];
  }

  T l2_norm() const {
    return l2_norm(std::integral_constant<bool, rank == 1>());
  }

private:
  T m_value[size]; ///< Data array

  T l2_norm(const std::true_type &) const {
    T norm = 0.0;
    for (std::size_t i = 0; i < size; ++i) {
      norm += m_value[i] * m_value[i];
    }
    return Kokkos::sqrt(norm);
  }

  T l2_norm(const std::false_type &) const {
    static_assert(rank == 1, "l2_norm is only implemented for 1-D arrays");
    return 0.0;
  }
};

} // namespace impl
} // namespace datatype
} // namespace specfem
