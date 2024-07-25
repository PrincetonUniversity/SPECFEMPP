#pragma once

namespace specfem {

namespace iterator {
namespace impl {
template <bool UseSIMD> struct range_index_type;

template <> struct range_index_type<false> {
  specfem::point::assembly_index index;

  KOKKOS_INLINE_FUNCTION
  range_index_type(const specfem::point::assembly_index index) : index(index) {}
};

template <> struct range_index_type<true> {
  specfem::point::simd_assembly_index index;

  KOKKOS_INLINE_FUNCTION
  range_index_type(const specfem::point::simd_assembly_index index)
      : index(index) {}
};
} // namespace impl

template <typename SIMDType> struct range {
private:
  int starting_index;
  int number_points;

  constexpr static bool using_simd = SIMDType::using_simd;
  constexpr static int simd_size = SIMDType::size();

  KOKKOS_INLINE_FUNCTION
  range(const int starting_index, const int number_points, std::true_type)
      : starting_index(starting_index),
        number_points((number_points < simd_size) ? number_points : simd_size) {
  }

  KOKKOS_INLINE_FUNCTION
  range(const int starting_index, const int number_points, std::false_type)
      : starting_index(starting_index), number_points(number_points) {}

  KOKKOS_INLINE_FUNCTION
  impl::range_index_type<false> operator()(const int i, std::false_type) const {
    return impl::range_index_type<false>(
        specfem::point::assembly_index{ starting_index });
  }

  KOKKOS_INLINE_FUNCTION
  impl::range_index_type<true> operator()(const int i, std::true_type) const {
    return impl::range_index_type<true>(
        specfem::point::simd_assembly_index{ starting_index, number_points });
  }

public:
  KOKKOS_INLINE_FUNCTION
  range() = default;

  KOKKOS_INLINE_FUNCTION
  range(const int starting_index, const int number_points)
      : range(starting_index, number_points,
              std::integral_constant<bool, using_simd>()) {}

  KOKKOS_INLINE_FUNCTION
  impl::range_index_type<using_simd> operator()(const int i) const {
    return operator()(i, std::integral_constant<bool, using_simd>());
  }
};
} // namespace iterator

namespace policy {

template <typename ParallelConfig, typename ExecSpace>
struct range : Kokkos::RangePolicy<ExecSpace> {
public:
  using simd = typename ParallelConfig::simd;
  constexpr static int simd_size = simd::size();
  constexpr static bool using_simd = simd::using_simd;

  using PolicyType = Kokkos::RangePolicy<ExecSpace>;

  KOKKOS_FUNCTION
  range() = default;

  KOKKOS_FUNCTION range(const int range_size)
      : PolicyType(0, range_size / simd_size + (range_size % simd_size != 0)),
        range_size(range_size) {}

  inline const PolicyType &get_policy() const { return *this; }

  KOKKOS_FUNCTION auto range_iterator(const int range_index) const {
    const int starting_index = range_index * simd_size;
    const int number_elements = (range_index + simd_size < range_size)
                                    ? simd_size
                                    : range_size - range_index;

    return specfem::iterator::range<simd>(starting_index, number_elements);
  }

private:
  int range_size;
};
} // namespace policy
} // namespace specfem
