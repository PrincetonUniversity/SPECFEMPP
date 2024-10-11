#include "point/coordinates.hpp"
#include "point/coordinates.tpp"

// explicit instantiation of the distance function for dim2
template KOKKOS_FUNCTION type_real specfem::point::distance(
    const specfem::point::global_coordinates<specfem::dimension::type::dim2> &,
    const specfem::point::global_coordinates<specfem::dimension::type::dim2> &);
