#pragma once

#include "specfem/assembly.hpp"

namespace specfem::compute::impl {

template <int NGLL>
struct StackStoredChunkFaceArray
    : public specfem::datatype::RegisterArray<
          typename specfem::datatype::simd<type_real,
                                           false /*using_simd*/>::datatype,
          Kokkos::extents<std::size_t, 1, NGLL, NGLL, 1>, Kokkos::layout_left> {

  constexpr static bool using_simd =
      false; ///< Use SIMD datatypes for the array. If false,
             ///< std::is_same<value_type, base_type>::value is true
  using base_type = specfem::datatype::RegisterArray<
      typename specfem::datatype::simd<type_real, using_simd>::datatype,
      Kokkos::extents<std::size_t, 1, NGLL, NGLL, 1>, Kokkos::layout_left>;
  using simd = specfem::datatype::simd<type_real, using_simd>; ///< SIMD data
                                                               ///< type
  using value_type =
      typename base_type::value_type;  ///< Value type used to store
                                       ///< the elements of the array
  constexpr static int components = 1; ///< Number of components of the
                                       ///< vector
  static constexpr int ngll = NGLL;

  using base_type::base_type;
  constexpr static auto accessor_type =
      specfem::datatype::AccessorType::chunk_face;
};

template <int NGLL, typename Tags>
void compute_coupling_conjugate_integral_nonconforming(
    const specfem::assembly::assembly<specfem::element::dimension_tag::dim3>
        &assembly)
  requires(Tags::dimension_tag == specfem::element::dimension_tag::dim3);

template <int NGLL, typename Tags>
void compute_coupling_conjugate_integral_nonconforming(
    const specfem::assembly::assembly<specfem::element::dimension_tag::dim2>
        &assembly)
  requires(Tags::dimension_tag == specfem::element::dimension_tag::dim2);

} // namespace specfem::compute::impl
