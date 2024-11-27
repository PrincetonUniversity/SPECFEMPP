#include "test_fixture.hpp"

template <>
KOKKOS_FUNCTION specfem::point::index<specfem::dimension::type::dim2, true>
get_index<true>(const int ielement, const int num_elements, const int iz,
                const int ix) {
  return specfem::point::simd_index<specfem::dimension::type::dim2>(
      ielement, num_elements, iz, ix);
}

template <>
KOKKOS_FUNCTION specfem::point::index<specfem::dimension::type::dim2, false>
get_index<false>(const int ielement, const int num_elements, const int iz,
                 const int ix) {
  return specfem::point::index<specfem::dimension::type::dim2>(ielement, iz,
                                                               ix);
}
