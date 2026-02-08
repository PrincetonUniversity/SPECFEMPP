#include "wavefield_reader.hpp"
#include "specfem/io.hpp"

// Explicit instantiation
template class specfem::periodic_tasks::wavefield_reader<
    specfem::element::dimension_tag::dim2, specfem::io::HDF5>;

template class specfem::periodic_tasks::wavefield_reader<
    specfem::element::dimension_tag::dim2, specfem::io::ASCII>;

template class specfem::periodic_tasks::wavefield_reader<
    specfem::element::dimension_tag::dim2, specfem::io::NPY>;

template class specfem::periodic_tasks::wavefield_reader<
    specfem::element::dimension_tag::dim2, specfem::io::NPZ>;
