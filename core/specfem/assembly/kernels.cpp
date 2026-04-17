#include "specfem/assembly/kernels.hpp"

template struct specfem::assembly::kernels<
    specfem::element::dimension_tag::dim2>;
template struct specfem::assembly::kernels<
    specfem::element::dimension_tag::dim3>;
