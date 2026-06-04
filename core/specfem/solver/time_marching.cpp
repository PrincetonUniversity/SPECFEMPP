#include "specfem/solver/time_marching.hpp"
#include "specfem/element.hpp"
#include "specfem/enums.hpp"
#include "specfem/solver/time_marching.tpp"

// Explcit template instantiation

// 2D instantiations
template class specfem::solver::time_marching<
    specfem::simulation::type::forward, specfem::element::dimension_tag::dim2,
    5>;

template class specfem::solver::time_marching<
    specfem::simulation::type::forward, specfem::element::dimension_tag::dim2,
    8>;

template class specfem::solver::time_marching<
    specfem::simulation::type::combined, specfem::element::dimension_tag::dim2,
    5>;

template class specfem::solver::time_marching<
    specfem::simulation::type::combined, specfem::element::dimension_tag::dim2,
    8>;

// 3D instantiations
template class specfem::solver::time_marching<
    specfem::simulation::type::forward, specfem::element::dimension_tag::dim3,
    5>;

// template class specfem::solver::time_marching<
//     specfem::simulation::type::forward,
//     specfem::element::dimension_tag::dim3, 8>;

template class specfem::solver::time_marching<
    specfem::simulation::type::combined, specfem::element::dimension_tag::dim3,
    5>;
