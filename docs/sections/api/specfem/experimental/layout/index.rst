Hierarchical layouts
====================

The experimental ``specfem::experimental::layout`` module maps recursive logical
coordinates to element offsets independently of storage ownership. Include
``specfem/experimental/layout/layout.hpp`` for mappings or
``specfem/experimental/layout/view.hpp`` for borrowed element access. Existing
``DomainView`` types and storage conventions remain unchanged.

Construction and indexing
-------------------------

.. code-block:: cpp

    namespace hl = specfem::experimental::layout;
    const auto mapping = hl::make_layout(
        hl::make_tuple(hl::make_tuple(hl::c<4>, 3), hl::c<2>),
        hl::make_tuple(hl::make_tuple(hl::c<1>, 16), hl::c<4>));
    const auto scalar = hl::make_tuple(5, 1);
    const auto nested = hl::make_tuple(hl::make_tuple(1, 1), 1);
    // Both coordinates have offset 21.
    const auto offset = mapping.offset(scalar);

    // backing points to 48 accessible elements in this example.
    const auto view = hl::make_view(backing, 48, mapping);
    if (mapping.valid(nested)) {
      view(nested) = value;
    }

``make_tuple`` constructs device-callable recursive metadata. ``c<N>`` preserves
an integral value in its type. Runtime integers and constants can occur
independently at every shape or stride leaf. Each ``make_tuple`` argument must
satisfy ``TupleElement``: a non-boolean integer, an integral constant of one, or
a nested tuple. Other types fail at the ``make_tuple`` call. Constants and fully
static subtuples occupy no storage, so a tuple's size is the size of its runtime
leaves. ``get<I>`` returns a copy of a tuple child without erasing its static
type. Hierarchy topology and rank are compile-time properties; nesting has no
fixed depth limit beyond compiler resources.

``is_tuple_v<T>`` identifies metadata tuples constructed by ``make_tuple``.
``tuple_size_v<T>`` counts a tuple's immediate children, including zero for an
empty tuple. Both queries ignore const and reference qualifiers. Query arity only
after establishing that the node is a tuple. Leaves include runtime integers and
integral constants. Together with ``get<I>``, these traits let generic adapters
walk the hierarchy without naming an implementation type. They do not add a
standard-library tuple protocol.

``rank()`` counts top-level modes. ``extent<I>()`` multiplies the leaves in that
mode.

A query is static when everything it depends on is static, as in CuTe.
``extent<I>()`` returns ``std::integral_constant<std::size_t, N>`` when the
entire mode is static, ``envelope_size()`` when the entire shape is static, and
``offset(coord)`` when the shape, strides, and coordinate are all static.
Otherwise each returns ``std::size_t``. ``required_span_size()`` always returns
``std::size_t``, because it depends on the runtime enabled flag. Generic code
that modifies a result should store it in ``std::size_t``.
``shape()`` and ``strides()`` retain the complete recursive metadata. A nested
mode need not have one affine stride.

Coordinates are tuples at the top level. A scalar can replace a nested subtree;
it decomposes recursively with the first leaf fastest. Explicit nested
coordinates follow the shape topology. The scalar must be less than the
subtree's full extent before decomposition. Modulo does not make an invalid
scalar coordinate valid. Callers establish validity before element access.

Domain and storage sizes
------------------------

``valid(coord)`` queries logical membership and returns false for negative or
out-of-bounds coordinates. ``offset(coord)`` requires a valid coordinate and
returns an offset in elements relative to the original backing pointer.

``envelope_size()`` is the product of the shape leaves. It counts the rectangular
coordinate envelope independently of logical restrictions. ``empty()`` reports
whether there is any valid coordinate. Root layouts use rectangular bounds and
an optional ``enabled`` argument to ``make_layout``. Passing false disables the
entire logical domain. Future transformations must preserve all inherited
restrictions independently of offset metadata.

Construction diagnoses type-known backing-span errors at compile time, with or
without an enabled flag. No runtime flag can make such a layout valid. Backing
spans that depend on runtime values are validated only when the layout is
enabled, in debug execution or constant evaluation. A disabled mapping needs no
backing storage even if its runtime values would require an unrepresentable
span. Leaf and envelope-size checks still apply.

For a nonempty root layout, ``required_span_size()`` is one plus the sum of
``(extent - 1) * stride`` over the leaves. It includes holes, but not arbitrary
allocation padding. The example has envelope size 24, required span 40, and a
backing capacity of 48. Padding and holes never grant logical validity.

A zero extent makes the domain empty and the required span zero. Empty-domain
queries avoid division by zero and extent subtraction underflow. A rank-zero
layout uses empty shape and stride tuples. It has one valid empty coordinate,
offset zero, and required span one, unless disabled. A disabled rank-zero
layout requires zero storage while retaining an envelope size of one.

Borrowed storage and future transformations
-------------------------------------------

A borrowed view retains the original pointer, capacity in elements, and layout.
It allocates nothing and copies no elements. The caller must keep the storage
alive, use it only in an execution space that can access its memory, and wait
for asynchronous work before releasing it. ``const T *`` produces const element
references. A const view object over ``T *`` still permits writes, as with a
Kokkos handle. A view over ``T`` converts implicitly to a view over
``const T`` with the same mapping. Conversions that would change the element
size, such as derived to base, are rejected. A null pointer is permitted only for an empty logical domain.

The internal ``rebind_layout`` operation gives future transformations one place
to retain the original backing handle and capacity. Slice displacement must
remain relative to that original pointer. Rebinding checks backing capacity in
debug builds, but does not establish parent-domain correspondence. Each
transformation must establish that property. Rebinding is not a public reshape
operation. General transformations and managed Kokkos ownership adapters are
separate implementation steps.

Preconditions and diagnostics
-----------------------------

Shape and stride topology must match. Leaves must be nonnegative integers
representable in ``std::size_t``. Runtime arithmetic, bounds, nonnull storage,
and backing capacity receive descriptive host/device diagnostics in builds
without ``NDEBUG``. Release builds omit runtime validation; violating runtime
preconditions has undefined behavior. Type-known invalid configurations receive
compile-time diagnostics in either build mode. Invalid operations during
constant evaluation fail compilation.

Inputs must describe an injective mapping on valid coordinates. This indexing
module does not yet certify uniqueness. Broadcasts and overlapping valid
coordinates are prohibited preconditions, not supported layout families.
Negative strides and general composed offset functions are excluded.

Reference
---------

.. doxygenclass:: specfem::experimental::layout::Layout
    :members:

.. doxygenclass:: specfem::experimental::layout::View
    :members:

.. doxygenfunction:: specfem::experimental::layout::make_layout(Shape, Strides)

.. doxygenfunction:: specfem::experimental::layout::make_layout(Shape, Strides, bool)

.. doxygenfunction:: specfem::experimental::layout::make_view

.. doxygenfunction:: specfem::experimental::layout::make_tuple

.. doxygenfunction:: specfem::experimental::layout::get
