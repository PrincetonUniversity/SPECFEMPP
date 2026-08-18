.. _algorithms_inside_outside:

``specfem::algorithms::inside`` / ``outside``
=============================================

Reference-element containment predicates for located points. ``outside`` is
deliberately not the negation of ``inside``: a located point in the tolerance
band (1 < \|coord\| <= tolerance) is neither inside nor outside, and an
unlocated point (``ispec < 0``) is not inside and always counts as outside.

.. doxygenfunction:: specfem::algorithms::inside

.. doxygenfunction:: specfem::algorithms::outside
