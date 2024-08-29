#include "domain/impl/kernels.hpp"
#include "domain/impl/kernels.tpp"

namespace {

using static_5 =
    specfem::enums::element::quadrature::static_quadrature_points<5>;
using static_8 =
    specfem::enums::element::quadrature::static_quadrature_points<8>;

} // namespace

// Explicit template instantiation

template class specfem::domain::impl::kernels::kernels<
    specfem::wavefield::type::forward, specfem::dimension::type::dim2,
    specfem::element::medium_tag::elastic, static_5>;

template class specfem::domain::impl::kernels::kernels<
    specfem::wavefield::type::adjoint, specfem::dimension::type::dim2,
    specfem::element::medium_tag::elastic, static_5>;

template class specfem::domain::impl::kernels::kernels<
    specfem::wavefield::type::backward, specfem::dimension::type::dim2,
    specfem::element::medium_tag::elastic, static_5>;

template class specfem::domain::impl::kernels::kernels<
    specfem::wavefield::type::forward, specfem::dimension::type::dim2,
    specfem::element::medium_tag::acoustic, static_5>;

template class specfem::domain::impl::kernels::kernels<
    specfem::wavefield::type::adjoint, specfem::dimension::type::dim2,
    specfem::element::medium_tag::acoustic, static_5>;

template class specfem::domain::impl::kernels::kernels<
    specfem::wavefield::type::backward, specfem::dimension::type::dim2,
    specfem::element::medium_tag::acoustic, static_5>;

template class specfem::domain::impl::kernels::kernels<
    specfem::wavefield::type::forward, specfem::dimension::type::dim2,
    specfem::element::medium_tag::elastic, static_8>;

template class specfem::domain::impl::kernels::kernels<
    specfem::wavefield::type::adjoint, specfem::dimension::type::dim2,
    specfem::element::medium_tag::elastic, static_8>;

template class specfem::domain::impl::kernels::kernels<
    specfem::wavefield::type::backward, specfem::dimension::type::dim2,
    specfem::element::medium_tag::elastic, static_8>;

template class specfem::domain::impl::kernels::kernels<
    specfem::wavefield::type::forward, specfem::dimension::type::dim2,
    specfem::element::medium_tag::acoustic, static_8>;

template class specfem::domain::impl::kernels::kernels<
    specfem::wavefield::type::adjoint, specfem::dimension::type::dim2,
    specfem::element::medium_tag::acoustic, static_8>;

template class specfem::domain::impl::kernels::kernels<
    specfem::wavefield::type::backward, specfem::dimension::type::dim2,
    specfem::element::medium_tag::acoustic, static_8>;
