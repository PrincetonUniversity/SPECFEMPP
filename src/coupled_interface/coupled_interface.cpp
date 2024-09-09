#include "coupled_interface/coupled_interface.hpp"
#include "coupled_interface/coupled_interface.tpp"
#include "enumerations/dimension.hpp"
#include "enumerations/medium.hpp"

template class specfem::coupled_interface::coupled_interface<
    specfem::wavefield::type::forward, specfem::dimension::type::dim2,
    specfem::element::medium_tag::elastic,
    specfem::element::medium_tag::acoustic>;

template class specfem::coupled_interface::coupled_interface<
    specfem::wavefield::type::forward, specfem::dimension::type::dim2,
    specfem::element::medium_tag::acoustic,
    specfem::element::medium_tag::elastic>;

template class specfem::coupled_interface::coupled_interface<
    specfem::wavefield::type::backward, specfem::dimension::type::dim2,
    specfem::element::medium_tag::elastic,
    specfem::element::medium_tag::acoustic>;

template class specfem::coupled_interface::coupled_interface<
    specfem::wavefield::type::backward, specfem::dimension::type::dim2,
    specfem::element::medium_tag::acoustic,
    specfem::element::medium_tag::elastic>;

template class specfem::coupled_interface::coupled_interface<
    specfem::wavefield::type::adjoint, specfem::dimension::type::dim2,
    specfem::element::medium_tag::elastic,
    specfem::element::medium_tag::acoustic>;

template class specfem::coupled_interface::coupled_interface<
    specfem::wavefield::type::adjoint, specfem::dimension::type::dim2,
    specfem::element::medium_tag::acoustic,
    specfem::element::medium_tag::elastic>;
