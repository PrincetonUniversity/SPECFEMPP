.. _specfem_point_conforming_interface:

``specfem::point::conforming_interface``
========================================

.. doxygenstruct:: specfem::point::conforming_interface
    :members:

2D Specialization
-----------------

.. doxygenstruct:: specfem::point::conforming_interface< specfem::element::dimension_tag::dim2, InterfaceTag, BoundaryTag >
    :members:

3D Specialization
-----------------

.. doxygenstruct:: specfem::point::conforming_interface< specfem::element::dimension_tag::dim3, InterfaceTag, BoundaryTag >
    :members:

Orientation and surface measure
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For weakly conforming fluid-solid faces, ``self`` is the medium receiving the
coupling contribution. At every face GLL point, ``face_normal`` stores
:math:`J_s\mathbf{n}_{self}`, where :math:`J_s` is the surface Jacobian and
:math:`\mathbf{n}_{self}` is the outward unit normal of that element.
``face_factor`` stores only the product of the two tangential quadrature weights.
Their product is the oriented surface integration measure. Neither quantity
should be replaced by a face-wide constant on a curved element.

The normals have opposite senses for the two coupling directions:

.. list-table:: Radial orientation on core interfaces
   :header-rows: 1

   * - Interface
     - Acoustic self (outer core)
     - Elastic self
   * - CMB
     - Outward, :math:`+\hat{\mathbf r}`
     - Inward, :math:`-\hat{\mathbf r}` (mantle)
   * - ICB
     - Inward, :math:`-\hat{\mathbf r}`
     - Outward, :math:`+\hat{\mathbf r}` (inner core)

These directions are approximate for deformed geometry, such as an ellipsoid.
Both kernels use their own self normal with a positive coefficient. With the
acoustic potential convention :math:`p=-\ddot\chi`, the solid traction is
:math:`-p\mathbf n_{solid}=\ddot\chi\mathbf n_{solid}`. The fluid receives
:math:`\mathbf n_{fluid}\cdot\mathbf u_{solid}`. An additional sign based on
CMB versus ICB would therefore be incorrect. The transfer function supplies the
correspondence between the two faces' GLL orderings.

Globe mesh setup derives these interfaces from medium-contrast adjacency edges,
which are promoted to ``weakly_conforming``. The database CMB and ICB lists
cross-check that derived set; they do not create a second set of interfaces.
The two media retain distinct global degrees of freedom, even at coincident
physical points.

Validation geometry
~~~~~~~~~~~~~~~~~~~

``GlobeFluidSolidInterfaces`` tests CMB and ICB separately, in both directions,
using the production reader, assembly mesh, Jacobians, interface containers and
pointwise coupling kernels. Surface metrics are checked against cross products
of independently differentiated physical coordinates. Signed radial alignment
is checked within 0.005 radians plus 10 percent of the face angular diameter,
allowing ellipticity and hex27 interpolation error on the coarser ICB. The tests
also require normals to vary within faces and the two media's global index sets
to be disjoint, including edge and corner nodes.

The committed ``GlobalSmallMesh`` fixture is an ellipsoidal, single 90-degree
chunk. Its open CMB/ICB patches have nonzero integrals of the normal, so the
closure test compares each patch's vector area to the independent rim integral
:math:`\frac12\oint\mathbf r\times d\mathbf r`. Equivalently, radial sidewalls
to the origin close the patch and give zero total vector area. Pointwise signed
alignment is essential: a globally reversed closed surface still has zero
integral of its normal.

A synthetic variant uses the database's undeformed anchors as physical geometry
and checks each spherical patch against :math:`4\pi R^2/6` within 0.1 percent.
Here :math:`R` is the mean radius of the assembled face centres. This variant
explicitly constructs a different physical mesh; production Jacobians always
use the final deformed coordinates. It avoids applying a whole-sphere area
formula to an ellipsoidal or partial surface. Full six-chunk database coverage
remains separate from these single-partition tests.
