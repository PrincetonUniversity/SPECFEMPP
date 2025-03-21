#include "IO/kernel/writer.hpp"
#include "IO/ASCII/ASCII.hpp"
#include "IO/HDF5/HDF5.hpp"
#include "IO/impl/medium_writer.tpp"
#include "IO/kernel/writer.tpp"

// Explicit instantiation

template class specfem::io::kernel_writer<
    specfem::io::HDF5<specfem::io::write> >;

template class specfem::io::kernel_writer<
    specfem::io::ASCII<specfem::io::write> >;
