
.. _injection_api:

``specfem::injection``
======================

.. doxygennamespace:: specfem::injection
    :desc-only:

The injection component drives a local simulation with an externally computed
incident wavefield. Injection is a *pluggable method*: a host-polymorphic
provider produces the injected field, which it exposes to the stiffness kernel
through a uniform device-resident frame buffer. The frequency-wavenumber (FK)
solver is the first provider; additional methods are added as new
:cpp:class:`specfem::injection::injection_provider` subclasses without changing
the interface.


``specfem::injection::injection_provider``
------------------------------------------

.. doxygenclass:: specfem::injection::injection_provider
    :members:


``specfem::injection::InjectionFrameBuffer``
--------------------------------------------

.. doxygenclass:: specfem::injection::InjectionFrameBuffer
    :members:


``specfem::injection::fk_provider``
-----------------------------------

.. doxygenclass:: specfem::injection::fk_provider
    :members:


.. toctree::
   :maxdepth: 1

   fk/index
