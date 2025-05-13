
.. _libraries:

IO Libraries
============

SPECFEM++ IO libraries provide a set of modules that can be used to read and write ``Kokkos::View`` objects to and from disk.

The snippet below shows how to use these modules to write and read a Kokkos::View to and from disk.

.. code:: cpp

    // Library needs to have write access
    template<typename Library>
    void write(const Kokkos::View<type_real> data) {
        typename Library::File file("filename");
        const auto dataset = file.CreateDataset("data", data);
        dataset.write();
        return;
    };

    // Library needs to have read access
    template<typename Library>
    Kokkos::View<type_real> read() {
        Kokkos::View<type_real> data;
        typename Library::File file("filename");
        const auto dataset = file.OpenDataset("data", data);
        dataset.read();
        return data;
    };

    int main() {
        Kokkos::View<type_real> data("data", 10);
        write<specfem::io::HDF5<specfem::io::write>>(data);
        const auto data_read = read<specfem::io::HDF5<specfem::io::read>>();
        return 0;
    }

.. toctree::
    :maxdepth: 1

    sources/index
    mesh/index
    ASCII/index
    HDF5/index
