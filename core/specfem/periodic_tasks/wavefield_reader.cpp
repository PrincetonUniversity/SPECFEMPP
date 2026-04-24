#include "wavefield_reader.hpp"
#include "specfem/io.hpp"

// Explicit instantiation
template class specfem::periodic_tasks::wavefield_reader<
    specfem::element::dimension_tag::dim2, specfem::io_backends::HDF5>;

template class specfem::periodic_tasks::wavefield_reader<
    specfem::element::dimension_tag::dim2, specfem::io_backends::ASCII>;

template class specfem::periodic_tasks::wavefield_reader<
    specfem::element::dimension_tag::dim2, specfem::io_backends::NPY>;

template class specfem::periodic_tasks::wavefield_reader<
    specfem::element::dimension_tag::dim2, specfem::io_backends::NPZ>;
