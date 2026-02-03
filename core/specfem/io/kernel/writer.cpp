#include "specfem/io/kernel/writer.hpp"
#include "specfem/io/ADIOS2/ADIOS2.hpp"
#include "specfem/io/ASCII/ASCII.hpp"
#include "specfem/io/HDF5/HDF5.hpp"
#include "specfem/io/NPY/NPY.hpp"
#include "specfem/io/NPZ/NPZ.hpp"

#include "specfem/io/impl/medium_writer.tpp"
#include "specfem/io/kernel/writer.tpp"

// Explicit instantiation

template class specfem::io::kernel_writer<
    specfem::io::HDF5<specfem::io::write> >;

template class specfem::io::kernel_writer<
    specfem::io::ADIOS2<specfem::io::write> >;

template class specfem::io::kernel_writer<
    specfem::io::ASCII<specfem::io::write> >;

template class specfem::io::kernel_writer<
    specfem::io::NPY<specfem::io::write> >;

template class specfem::io::kernel_writer<
    specfem::io::NPZ<specfem::io::write> >;
