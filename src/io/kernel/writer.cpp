#include "io/kernel/writer.hpp"
#include "io/ASCII/ASCII.hpp"
#include "io/HDF5/HDF5.hpp"
#include "io/impl/medium_writer.tpp"
#include "io/kernel/writer.tpp"

// Explicit instantiation

template class specfem::io::kernel_writer<
    specfem::io::HDF5<specfem::io::write> >;

template class specfem::io::kernel_writer<
    specfem::io::ASCII<specfem::io::write> >;
