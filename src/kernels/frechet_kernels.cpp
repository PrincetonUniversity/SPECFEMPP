#include "kernels/frechet_kernels.hpp"
#include "kernels/frechet_kernels.tpp"

// Explicit template instantiation
template class specfem::kernels::frechet_kernels<specfem::dimension::type::dim2,
                                                 5>;

template class specfem::kernels::frechet_kernels<specfem::dimension::type::dim2,
                                                 8>;
