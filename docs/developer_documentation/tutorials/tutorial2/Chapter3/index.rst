
.. _tutorial2_Chapter3:

Chapter 3: Reading the forward wavefield from disk
==================================================

Now that we have written the forward wavefield to disk, let's read it back into the buffer field.

.. code:: cpp

    template <typename InputLibrary>
    class reader {
    public:
        using IO_Library = InputLibrary;

        /**
        * Create a new wavefield reader
        * @param filename Name of the file to read from
        */
        reader(
            const std::string filename,
            simulation_field<buffer_tag> &buffer) :
                filename(filename), buffer(buffer){};

        /**
        * Read the wavefield from disk
        */
        void read();

    private:
        std::string filename;
        simulation_field<buffer_tag> buffer;
    };

Again, similar to the writer class, the reader class is templated on the input library. The ``InputLibrary`` needs to be same as the one used to write the wavefield. Now let's implement the read function.

.. code:: cpp

    template <typename InputLibrary>
    void reader<InputLibrary>::read() {

        typename InputLibrary::File file(filename + "/ForwardWavefield");

        typename InputLibrary::Group elastic = file.openGroup("/Elastic");
        typename InputLibrary::Group acoustic = file.openGroup("/Acoustic");

        elastic.openDataset("Displacement", buffer.elastic.h_field).read();
        elastic.openDataset("Velocity", buffer.elastic.h_field_dot).read();
        elastic.openDataset("Acceleration", buffer.elastic.h_field_dot_dot).read();

        acoustic.openDataset("Potential", buffer.acoustic.h_field).read();
        acoustic.openDataset("PotentialDot", buffer.acoustic.h_field_dot).read();
        acoustic.openDataset("PotentialDotDot", buffer.acoustic.h_field_dot_dot)
            .read();

        return;
    }

SPECFEM++ Implementation Details
--------------------------------

The actual SPECFEM++ implementation details can be found :ref:`here <IO_wavefield_reader>`.
