#include "writer/kernel.hpp"
#include "IO/ASCII/ASCII.hpp"
#include "IO/HDF5/HDF5.hpp"
#include "writer/kernel.tpp"

// Explicit instantiation

template class specfem::writer::kernel<specfem::IO::HDF5<specfem::IO::write> >;

template class specfem::writer::kernel<specfem::IO::ASCII<specfem::IO::write> >;
