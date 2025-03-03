#include "IO/kernel/writer.hpp"
#include "IO/ASCII/ASCII.hpp"
#include "IO/HDF5/HDF5.hpp"
#include "IO/kernel/writer.tpp"

// Explicit instantiation

template class specfem::IO::kernel_writer<
    specfem::IO::HDF5<specfem::IO::write> >;

template class specfem::IO::kernel_writer<
    specfem::IO::ASCII<specfem::IO::write> >;
