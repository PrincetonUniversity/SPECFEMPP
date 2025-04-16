#include "periodic_tasks/wavefield_reader.hpp"
#include "io/ASCII/ASCII.hpp"
#include "io/HDF5/HDF5.hpp"

// Explicit instantiation
template class specfem::periodic_tasks::wavefield_reader<
    specfem::io::HDF5<specfem::io::read> >;

template class specfem::periodic_tasks::wavefield_reader<
    specfem::io::ASCII<specfem::io::read> >;
