#include "IO/wavefield/writer.hpp"
#include "IO/ASCII/ASCII.hpp"
#include "IO/HDF5/HDF5.hpp"
#include "IO/wavefield/writer.tpp"

// Explicit instantiation

template class specfem::io::wavefield_writer<
    specfem::io::HDF5<specfem::io::write> >;

template class specfem::io::wavefield_writer<
    specfem::io::ASCII<specfem::io::write> >;
