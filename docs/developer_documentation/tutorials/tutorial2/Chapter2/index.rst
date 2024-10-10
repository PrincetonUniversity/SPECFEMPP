
.. _tutorial2_Chapter2:

Chapter 2: Writing the forward wavefield to disk
================================================

Next, let's create a writer class to write the forward wavefield to disk.

.. code:: cpp

    template <typename OutputLibrary>
    class writer {
    public:
        using IO_Library = OutputLibrary;

        /**
        * Create a new wavefield writer
        * @param filename Name of the file to write to (Creates a folder when writing in ASCII)
        */
        writer(
            const std::string filename,
            simulation_field<forward_tag> &forward) :
                filename(filename), forward(forward){};

        /**
        * Write the wavefield to disk
        */
        void write();

    private:
        std::string filename;
        simulation_field<forward_tag> forward;
    };

The writer class is templated on the output library. Since, all the IO libraries have the same interface, we can develop a writer that is agnostic to the output library. The ``OutputLibrary`` can then be either determined at runtime. Now let's implement the write function.

.. code:: cpp

    template <typename OutputLibrary>
    void writer<OutputLibrary>::write() {

        typename OutputLibrary::File file(filename + "/ForwardWavefield");

        typename OutputLibrary::Group elastic = file.createGroup("/Elastic");
        typename OutputLibrary::Group acoustic = file.createGroup("/Acoustic");

        elastic.createDataset("Displacement", forward.elastic.h_field).write();
        elastic.createDataset("Velocity", forward.elastic.h_field_dot).write();
        elastic.createDataset("Acceleration", forward.elastic.h_field_dot_dot).write();

        acoustic.createDataset("Potential", forward.acoustic.h_field).write();
        acoustic.createDataset("PotentialDot", forward.acoustic.h_field_dot).write();
        acoustic.createDataset("PotentialDotDot", forward.acoustic.h_field_dot_dot)
            .write();

        return;
    }

SPECFEM++ Implementation Details
--------------------------------

The actual SPECFEM++ implementation details can be found :ref:`here <IO_write_wavefield>`.
