#include "coupled_interface/coupled_interface.hpp"
#include "coupled_interface/coupled_interface.tpp"
#include "coupled_interface/impl/edge/edge.hpp"
#include "coupled_interface/impl/edge/elastic_acoustic/acoustic_elastic.tpp"
#include "coupled_interface/impl/edge/elastic_acoustic/elastic_acoustic.tpp"
#include "enumerations/dimension.hpp"
#include "enumerations/medium.hpp"

// Explicit template instantiations

template class specfem::coupled_interface::impl::edges::edge<
    specfem::dimension::type::dim2, specfem::element::medium_tag::elastic,
    specfem::element::medium_tag::acoustic>;

template class specfem::coupled_interface::impl::edges::edge<
    specfem::dimension::type::dim2, specfem::element::medium_tag::acoustic,
    specfem::element::medium_tag::elastic>;

// Explicit template instantiations

template class specfem::coupled_interface::coupled_interface<
    specfem::dimension::type::dim2, specfem::element::medium_tag::elastic,
    specfem::element::medium_tag::acoustic>;

template class specfem::coupled_interface::coupled_interface<
    specfem::dimension::type::dim2, specfem::element::medium_tag::acoustic,
    specfem::element::medium_tag::elastic>;
