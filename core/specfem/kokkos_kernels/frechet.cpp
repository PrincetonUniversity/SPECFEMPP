#include "frechet.hpp"

// Explicit template instantiation
template class specfem::kokkos_kernels::Frechet<specfem::dimension::type::dim2,
                                                5>;

template class specfem::kokkos_kernels::Frechet<specfem::dimension::type::dim2,
                                                8>;
