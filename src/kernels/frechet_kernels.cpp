#include "kernels/frechet_kernels.hpp"

// Explicit template instantiation
template class specfem::kernels::frechet_kernels<specfem::dimension::type::dim2,
                                                 5>;

template class specfem::kernels::frechet_kernels<specfem::dimension::type::dim2,
                                                 8>;
