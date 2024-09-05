
.. _tutorial2_Chapter1:

Understanding stored fields
===========================

Let's start by understanding how the fields are stored in SPECFEM++. The ``fields`` C++ struct stores 4 types of wavefields:

1. Forward wavefield : Primary wavefield during a forward simulation
2. Adjoint wavefield : Adjoint wavefield during a combined forward and adjoint simulation
3. Backward wavefield : Backward wavefield during a combined forward and adjoint simulation
4. Buffer field : Temporary field used to store wavefields read ahead of time

In this tutorial we will focus on writing the forward wavefield to disk and then reading that wavefield back from disk into the buffer field. Both these wavefields are instantiations of ``simulation_field`` datastruct outlined below. Refer :ref:`here <assembly_simulation_field>` for actual implementation details.

.. code:: cpp

    template <medium_tag MediumTag>
    struct field_impl {

        using ViewType = Kokkos::View<type_real ***>;

        ViewType field; // displacement field
        ViewType::HostMirror h_field;
        ViewType field_dot; // velocity field
        ViewType::HostMirror h_field_dot;
        ViewType field_dot_dot; // acceleration field
        ViewType::HostMirror h_field_dot_dot;
    }

    template <wavefield_tag Wavefield>
    struct simulation_field {
        field_impl<elastic_tag> elastic;
        field_impl<acoustic_tag> acoustic;
    }

    struct fields {
        simulation_field<forward_tag> forward;
        simulation_field<adjoint_tag> adjoint;
        simulation_field<backward_tag> backward;
        simulation_field<buffer_tag> buffer;
    }
