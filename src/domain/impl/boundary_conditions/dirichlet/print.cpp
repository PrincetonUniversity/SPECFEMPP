#include "domain/impl/boundary_conditions/dirichlet/print.tpp"

template std::string
specfem::domain::impl::boundary_conditions::print_boundary_tag<
    specfem::element::boundary_tag::acoustic_free_surface>();
