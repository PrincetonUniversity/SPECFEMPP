#include "test_fixture.hpp"

// Dim2 specializations
template <>
KOKKOS_FUNCTION specfem::point::index<specfem::element::dimension_tag::dim2, true>
get_index<true>(const int ielement, const int num_elements, const int iz,
                const int ix) {
  return specfem::point::index<specfem::element::dimension_tag::dim2, true>(
      ielement, num_elements, iz, ix);
}

template <>
KOKKOS_FUNCTION specfem::point::index<specfem::element::dimension_tag::dim2, false>
get_index<false>(const int ielement, const int num_elements, const int iz,
                 const int ix) {
  return specfem::point::index<specfem::element::dimension_tag::dim2>(ielement, iz,
                                                               ix);
}

// Dim3 specializations
template <>
KOKKOS_FUNCTION specfem::point::index<specfem::element::dimension_tag::dim3, true>
get_index<true>(const int ielement, const int num_elements, const int iz, const int iy,
                const int ix) {
  return specfem::point::index<specfem::element::dimension_tag::dim3, true>(
      ielement, num_elements, iz, iy, ix);
}

template <>
KOKKOS_FUNCTION specfem::point::index<specfem::element::dimension_tag::dim3, false>
get_index<false>(const int ielement, const int num_elements, const int iz, const int iy,
                 const int ix) {
  return specfem::point::index<specfem::element::dimension_tag::dim3>(ielement, iz, iy,
                                                               ix);
}
