#include "domain/impl/boundary_conditions/none/print.tpp"

template std::string
specfem::domain::impl::boundary_conditions::print_boundary_tag<
    specfem::element::boundary_tag::none>();
