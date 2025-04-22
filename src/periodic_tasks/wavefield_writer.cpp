#include "periodic_tasks/wavefield_writer.hpp"
#include "io/ASCII/ASCII.hpp"
#include "io/HDF5/HDF5.hpp"

// Explicit instantiation
template class specfem::periodic_tasks::wavefield_writer<
    specfem::io::HDF5<specfem::io::write> >;

template class specfem::periodic_tasks::wavefield_writer<
    specfem::io::ASCII<specfem::io::write> >;
