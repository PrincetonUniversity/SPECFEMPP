#include "kernels/kernels.hpp"

// Explicit template instantiation
template class specfem::kernels::kernels<
    specfem::wavefield::type::forward, specfem::dimension::type::dim2,
    specfem::enums::element::quadrature::static_quadrature_points<5> >;

template class specfem::kernels::kernels<
    specfem::wavefield::type::adjoint, specfem::dimension::type::dim2,
    specfem::enums::element::quadrature::static_quadrature_points<5> >;

template class specfem::kernels::kernels<
    specfem::wavefield::type::backward, specfem::dimension::type::dim2,
    specfem::enums::element::quadrature::static_quadrature_points<5> >;

// template class kernels<specfem::wavefield::type::forward,
// specfem::dimension::type::dim3,
// specfem::enums::element::quadrature::static_quadrature_points<8>>;

// template class kernels<specfem::wavefield::type::adjoint,
// specfem::dimension::type::dim3,
// specfem::enums::element::quadrature::static_quadrature_points<8>>;

// template class kernels<specfem::wavefield::type::backward,
// specfem::dimension::type::dim3,
// specfem::enums::element::quadrature::static_quadrature_points<8>>;
