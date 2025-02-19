#include "coupled_interface/coupled_interface.hpp"
#include "coupled_interface/coupled_interface.tpp"
#include "enumerations/dimension.hpp"
#include "enumerations/medium.hpp"

template class specfem::coupled_interface::coupled_interface<
    specfem::wavefield::simulation_field::forward,
    specfem::dimension::type::dim2, specfem::element::medium_tag::elastic_sv,
    specfem::element::medium_tag::acoustic>;

template class specfem::coupled_interface::coupled_interface<
    specfem::wavefield::simulation_field::forward,
    specfem::dimension::type::dim2, specfem::element::medium_tag::acoustic,
    specfem::element::medium_tag::elastic_sv>;

template class specfem::coupled_interface::coupled_interface<
    specfem::wavefield::simulation_field::backward,
    specfem::dimension::type::dim2, specfem::element::medium_tag::elastic_sv,
    specfem::element::medium_tag::acoustic>;

template class specfem::coupled_interface::coupled_interface<
    specfem::wavefield::simulation_field::backward,
    specfem::dimension::type::dim2, specfem::element::medium_tag::acoustic,
    specfem::element::medium_tag::elastic_sv>;

template class specfem::coupled_interface::coupled_interface<
    specfem::wavefield::simulation_field::adjoint,
    specfem::dimension::type::dim2, specfem::element::medium_tag::elastic_sv,
    specfem::element::medium_tag::acoustic>;

template class specfem::coupled_interface::coupled_interface<
    specfem::wavefield::simulation_field::adjoint,
    specfem::dimension::type::dim2, specfem::element::medium_tag::acoustic,
    specfem::element::medium_tag::elastic_sv>;
