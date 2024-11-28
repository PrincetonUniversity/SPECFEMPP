
#include "domain/domain.hpp"
#include "domain/domain.tpp"

// Explicit template instantiation

namespace {
using static_5 =
    specfem::enums::element::quadrature::static_quadrature_points<5>;
using static_8 =
    specfem::enums::element::quadrature::static_quadrature_points<8>;
} // namespace

template class specfem::domain::domain<
    specfem::wavefield::simulation_field::forward,
    specfem::dimension::type::dim2, specfem::element::medium_tag::elastic,
    static_5>;

template class specfem::domain::domain<
    specfem::wavefield::simulation_field::adjoint,
    specfem::dimension::type::dim2, specfem::element::medium_tag::elastic,
    static_5>;

template class specfem::domain::domain<
    specfem::wavefield::simulation_field::backward,
    specfem::dimension::type::dim2, specfem::element::medium_tag::elastic,
    static_5>;

template class specfem::domain::domain<
    specfem::wavefield::simulation_field::forward,
    specfem::dimension::type::dim2, specfem::element::medium_tag::acoustic,
    static_5>;

template class specfem::domain::domain<
    specfem::wavefield::simulation_field::adjoint,
    specfem::dimension::type::dim2, specfem::element::medium_tag::acoustic,
    static_5>;

template class specfem::domain::domain<
    specfem::wavefield::simulation_field::backward,
    specfem::dimension::type::dim2, specfem::element::medium_tag::acoustic,
    static_5>;

template class specfem::domain::domain<
    specfem::wavefield::simulation_field::forward,
    specfem::dimension::type::dim2, specfem::element::medium_tag::elastic,
    static_8>;

template class specfem::domain::domain<
    specfem::wavefield::simulation_field::adjoint,
    specfem::dimension::type::dim2, specfem::element::medium_tag::elastic,
    static_8>;

template class specfem::domain::domain<
    specfem::wavefield::simulation_field::backward,
    specfem::dimension::type::dim2, specfem::element::medium_tag::elastic,
    static_8>;

template class specfem::domain::domain<
    specfem::wavefield::simulation_field::forward,
    specfem::dimension::type::dim2, specfem::element::medium_tag::acoustic,
    static_8>;

template class specfem::domain::domain<
    specfem::wavefield::simulation_field::adjoint,
    specfem::dimension::type::dim2, specfem::element::medium_tag::acoustic,
    static_8>;

template class specfem::domain::domain<
    specfem::wavefield::simulation_field::backward,
    specfem::dimension::type::dim2, specfem::element::medium_tag::acoustic,
    static_8>;
