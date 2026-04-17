#include "specfem/assembly/properties.hpp"

template struct specfem::assembly::properties<
    specfem::element::dimension_tag::dim2>;
template struct specfem::assembly::properties<
    specfem::element::dimension_tag::dim3>;
