
.. _tutorial2_Chapter4:

Chapter 4: Instantiating the writer and reader
==============================================

Now that we have implemented the writer and reader classes, let's instantiate them in the main function with the HDF5 library.

.. code:: cpp

    int main() {
        const specfem::compute::assembly assembly;
        const auto& fields = assembly.fields;
        const auto& forward = fields.forward;
        const auto& buffer = fields.buffer;

        writer<specfem::io::HDF5<specfem::io::write>> writer("output", forward);
        writer.write();

        reader<specfem::io::HDF5<specfem::io::read>> reader("output", buffer);
        reader.read();

        // Deep copy the buffer into the backward field
        specfem::compute::deep_copy(fields.backward, fields.buffer);

        return 0;
    }

Similarly, you could instantiate the writer and reader with the ASCII library.

.. code:: cpp

    int main() {
        const specfem::compute::assembly assembly;
        const auto& fields = assembly.fields;
        const auto& forward = fields.forward;
        const auto& buffer = fields.buffer;

        writer<specfem::io::ASCII<specfem::io::write>> writer("output", forward);
        writer.write();

        reader<specfem::io::ASCII<specfem::io::read>> reader("output", buffer);
        reader.read();

        // Deep copy the buffer into the backward field
        specfem::compute::deep_copy(fields.backward, fields.buffer);

        return 0;
    }
