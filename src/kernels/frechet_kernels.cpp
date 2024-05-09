#include "kernels/frechet_kernels.hpp"
#include "frechet_derivatives/impl/frechet_element.tpp"

// Explicit template instantiation
template class specfem::kernels::frechet_kernels<
    5, specfem::dimension::type::dim2>;
