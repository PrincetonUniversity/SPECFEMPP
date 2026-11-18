Hierarchical layouts
====================

The experimental ``specfem::experimental::layout`` module maps recursive logical
coordinates to element offsets independently of storage ownership. Include
``specfem/experimental/layout/layout.hpp`` for mappings or
``specfem/experimental/layout/view.hpp`` for borrowed element access, or
``specfem/experimental/layout/kokkos.hpp`` for native adaptation.
``transform.hpp`` and ``subview.hpp`` add reorganization and selection. Existing
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
satisfy ``TupleElement``: a non-boolean integer, an integral constant of one, a
nested tuple, or a subview selector (``all`` or ``range``, see below). Other
types fail at the ``make_tuple`` call. Shapes, strides, and coordinates reject
selectors where they are used. Constants and fully
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
returns an offset in elements relative to the original backing pointer. The
offset is ``displacement()`` plus the sum of each coordinate leaf times its
stride. Root layouts have a static zero displacement, so their offsets need no
extra addition. Subviews carry the offset of their first selected element.

``envelope_size()`` is the product of the shape leaves. It counts the
rectangular coordinate envelope independently of logical restrictions. If
``empty()`` returns true, no coordinate is valid. Root layouts report emptiness
exactly, and reorganized mappings report their parent's answer. A subview can
return false even when it selects only invalid parent coordinates, such as tile
padding; anything reorganized from such a subview inherits that answer. Root
layouts use rectangular bounds and an optional ``enabled`` argument to
``make_layout``. Passing false disables the entire logical domain. Future
transformations must preserve all inherited restrictions independently of
offset metadata.

Construction diagnoses type-known backing-span errors at compile time, with or
without an enabled flag. No runtime flag can make such a layout valid. Backing
spans that depend on runtime values are validated only when the layout is
enabled, in debug execution or constant evaluation. A disabled mapping needs no
backing storage even if its runtime values would require an unrepresentable
span. Leaf and envelope-size checks still apply.

For a nonempty layout, ``required_span_size()`` is the displacement plus one
plus the sum of ``(extent - 1) * stride`` over the leaves. It includes holes,
but not arbitrary allocation padding. The example has envelope size 24, required
span 40, and a backing capacity of 48. Padding and holes never grant logical
validity.

A zero extent makes the domain empty and the required span zero. Empty-domain
queries avoid division by zero and extent subtraction underflow. A rank-zero
layout uses empty shape and stride tuples. It has one valid empty coordinate,
offset zero, and required span one, unless disabled. A disabled rank-zero
layout requires zero storage while retaining an envelope size of one.

Storage lifetime and transformations
------------------------------------

A borrowed view retains the original pointer, capacity in elements, and layout.
It allocates nothing and copies no elements. The caller must keep the storage
alive, use it only in an execution space that can access its memory, and wait
for asynchronous work before releasing it. ``const T *`` produces const element
references. A const view object over ``T *`` still permits writes, as with a
Kokkos handle. A view over ``T`` converts implicitly to a view over
``const T`` with the same mapping. Conversions that would change the element
size, such as derived to base, are rejected. A null pointer is permitted only for an empty logical domain.

The internal ``rebind_layout`` operation gives transformations one place to
retain the original backing handle and capacity. Subview displacements stay
relative to that original pointer, so the view never moves its pointer.
Rebinding checks backing capacity in debug builds, but does not establish
parent-domain correspondence. Each transformation must establish that property.
Rebinding is not a public reshape operation. General transformations are
separate implementation steps.

Native Kokkos storage
---------------------

Include ``specfem/experimental/layout/kokkos.hpp`` and pass a native view to
``make_view``. The adapter retains the exact native handle. Managed views and
managed subviews keep their allocation alive after the original handles leave
scope. Unmanaged views remain borrowed, including team scratch views.

.. code-block:: cpp

    Kokkos::View<int ***, Kokkos::LayoutLeft> data("data", 3, 8, 5);
    const auto slot = Kokkos::subview(data, 1, Kokkos::ALL, Kokkos::ALL);
    const auto view = hl::make_view(slot);
    Kokkos::parallel_for("write_slot", 8, KOKKOS_LAMBDA(const int j) {
      view(hl::make_tuple(j, 2)) = j;
    });
    Kokkos::fence();
    // data(1, j, 2) now contains j. Read it in an accessible execution space.

The adapter imports actual native extents and strides through ``make_layout``
certification. Static native extents retain their compile-time values; strides
come from the native handle even when the shape is static. A subview's offsets
start at its own ``data()`` pointer. Its accessible capacity is its ``span()``,
which includes native holes and padding. The adapter does not infer logical
validity from padding.

``make_view(native, mapping)`` retains the same native handle with an explicit
certified mapping. The mapping may have a different hierarchy or rank, but its
offsets must be relative to ``native.data()`` and its required span must fit
``native.span()``. This overload does not infer validity restrictions for a flat
backing allocation.

Supported native layouts are ``LayoutLeft``, ``LayoutRight`` and ``LayoutStride``
with ordinary element references. Atomic proxies and custom accessors/layouts
are not supported. Adapt production scratch wrappers through their native
``::type`` base explicitly:

.. code-block:: cpp

    using native_type = typename ScratchWrapper::type;
    const native_type &native_scratch = scratch;
    const auto view = hl::make_view(native_scratch);

A const handle over mutable elements still permits writes. A native view of
const elements gives const references. The adapter retains the native memory
space. Host construction and metadata queries do not dereference device memory.
Element access requires an execution space that can access that memory; debug
builds diagnose inaccessible access and release violations are undefined behavior.
Raw-pointer views carry no memory-space information.

Factories and supported transformations allocate no storage and fence no work.
Retain the managed adapter or another owner until asynchronous work completes.
Keep borrowed storage alive for that duration. A scratch adapter cannot extend
the team's scratch lifetime. Transformations retain the original handle through
the shared internal rewrapping operation, so they do not change these rules.
General mirrors and physical copies for transformed views remain outside this API.

Preconditions and diagnostics
-----------------------------

Shape and stride topology must match. Leaves must be nonnegative integers
representable in ``std::size_t``. Runtime arithmetic, bounds, nonnull storage,
and backing capacity receive descriptive host/device diagnostics in builds
without ``NDEBUG``. Release builds omit runtime validation; violating runtime
preconditions has undefined behavior. Type-known invalid configurations receive
compile-time diagnostics in either build mode. Invalid operations during
constant evaluation fail compilation.

Negative strides and general composed offset functions are excluded.

Uniqueness certification
------------------------

``make_layout`` certifies that distinct valid coordinates map to distinct
element offsets, so a layout introduces no hidden write aliasing. The rule
flattens the shape leaves, ignores leaves with extent one, and sorts the rest by
stride. Each stride must then be larger than the largest offset that the leaves
before it can reach, ``sum((extent - 1) * stride)``. Two coordinates that differ
then differ at a highest sorted leaf whose stride exceeds every lower
contribution, so their offsets differ.

The rule is sufficient, not necessary. It never accepts a broadcast (a zero
stride on a leaf with extent above one) or an overlap. It rejects some injective
layouts, for example shape ``(2, 3)`` with strides ``(3, 2)``. Gaps between
offsets are allowed, so padded storage certifies. Empty domains have no valid
coordinate and are always certified. Disabled domains skip runtime
certification, but a fully static descriptor must certify whatever the enabled
flag, because no runtime value can make it valid.

These layout families in current use are accepted. The certification tests
compare each one with the production code that produces it:

- contiguous ``Kokkos::LayoutLeft`` and ``Kokkos::LayoutRight`` views, including
  the CPU ``DomainView``;
- the CUDA (32 lanes) and HIP (64 lanes) ``DomainView`` storage tiles, written as
  ``((T, tiles), E1, ...) : ((1, T * E1 * ...), T, T * E1, ...)``, including
  the padded lanes of the last tile;
- static scratch views of element and chunk data in either array layout;
- strided Kokkos subviews with a fixed leading index or a leading range, as
  used by the forward displacement buffer, the wavefield helper, element
  intersections, and the chunked domain iterators.

A fully static descriptor that fails certification does not compile, with or
without an ``enabled`` flag. Constant evaluation of any failing descriptor
fails compilation. Other descriptors are checked on host and device in builds
without ``NDEBUG``.
In release builds no certification code runs, and constructing an uncertified
layout has undefined behavior.

Future transformations build derived layouts through an internal path that does
not certify again. Each transformation must guarantee that every valid derived
coordinate corresponds to exactly one valid parent coordinate with the same
offset, so the parent's certificate carries over. Certifying again would
wrongly reject some valid results. For example, tiling seven elements by four
leaves a padding slot whose offset can equal a valid element of the next row.

Reference
---------

.. doxygenclass:: specfem::experimental::layout::Layout
    :members:

.. doxygenclass:: specfem::experimental::layout::View
    :members:

.. doxygenfunction:: specfem::experimental::layout::make_layout(Shape, Strides)

.. doxygenfunction:: specfem::experimental::layout::make_layout(Shape, Strides, bool)

.. doxygenfunction:: specfem::experimental::layout::make_view(T *, std::size_t, const Mapping &)

.. doxygenfunction:: specfem::experimental::layout::make_view(const Kokkos::View<DataType, Properties...> &)

.. doxygenfunction:: specfem::experimental::layout::make_view(const Kokkos::View<DataType, Properties...> &, const Mapping &)

.. doxygenfunction:: specfem::experimental::layout::make_tuple

.. doxygenfunction:: specfem::experimental::layout::get

Permuting and regrouping
------------------------

Include ``specfem/experimental/layout/transform.hpp`` for ``permute``, ``group``
and ``ungroup``. Each accepts a mapping or an experimental ``View`` and returns
a value. These operations retain physical offsets, logical validity and backing
storage ownership. They allocate no element storage and copy no elements.
Managed views retain the original allocation after the parent objects leave
scope. Borrowed storage still requires the caller to keep storage alive until
asynchronous work completes. Element constness and memory-space access rules
remain those of the original view.

``ModePath<I...>`` selects a node by walking child positions. An empty path
selects the root tuple. Topology requests are compile-time values. Extents and
strides can remain runtime values without losing other leaves' static types.

* ``permute<Path, Order...>`` lists all immediate children of the selected tuple
  exactly once in their new order.
* ``group<Path, First, Count>`` replaces consecutive children with one tuple
  mode in the same order. Count must be positive. A count of one adds a level.
  Permute first to group nonconsecutive children.
* ``ungroup<Path>`` replaces a selected nonroot tuple child with its immediate
  children. Deeper nesting remains intact. An empty tuple child disappears.

.. code-block:: cpp

    #include "specfem/experimental/layout/transform.hpp"

    namespace hl = specfem::experimental::layout;
    const auto mapping = hl::make_layout(
        hl::make_tuple(hl::c<2>, 3), hl::make_tuple(hl::c<1>, 8));
    const auto grouped = hl::group<hl::ModePath<>, 0, 2>(mapping);
    // Shape ((2, 3)), strides ((1, 8)). Both coordinates have offset 17.
    const auto scalar_offset = grouped.offset(hl::make_tuple(5));
    const auto nested_offset =
        grouped.offset(hl::make_tuple(hl::make_tuple(1, 2)));
    const auto permuted = hl::permute<hl::ModePath<0>, 1, 0>(grouped);
    // Shape ((3, 2)). The original point now has scalar coordinate 5 as well.
    const auto offset = permuted.offset(hl::make_tuple(5));
    const auto flat = hl::ungroup<hl::ModePath<0>>(permuted);
    // Shape (3, 2). Coordinate (2, 1) still has offset 17.

Scalar coordinates always decompose with the first leaf fastest in the result's
hierarchy. A permutation within a mode can change the original point addressed
by a scalar coordinate. For example, scalar 1 in ``((2, 3))`` refers to original
coordinate ``(1, 0)``, while scalar 1 after permutation refers to ``(0, 1)``.
Grouping retains recursive strides. It does not require or approximate a single
affine stride for the resulting mode.

Call ``result.layout().valid(coord)`` before accessing a transformed view, or
``result.valid(coord)`` for a mapping. Negative and out-of-envelope coordinates
return false. Empty domains remain empty, and a rank-zero identity permutation
preserves a scalar's validity. Envelope size and required backing span equal
the parent's. Allocation padding never creates valid logical coordinates.

Malformed paths, invalid permutations, invalid grouping intervals, and ungrouping
a leaf or root fail compilation in every build mode. Type-detectable invalid
constant coordinates also fail compilation. Runtime-invalid accesses produce
diagnostics in debug builds and have undefined behavior in release builds.
Transforms and indexing are device-callable. A kernel can use runtime metadata,
but cannot select a different hierarchy topology at runtime.

The result's concrete mapping type is unspecified. Consumers use ``shape()``,
``strides()``, ``rank()``, ``extent<I>()``, ``envelope_size()``, ``empty()``,
``valid()``, ``offset()`` and ``required_span_size()``. Accepting that interface
lets subviews and later tiling retain their logical restrictions. Each mapping
also provides ``displacement()``, which reorganization forwards unchanged.
General composed offset mappings and transformed-view mirror/copy operations
remain outside this interface.

.. doxygenstruct:: specfem::experimental::layout::ModePath

.. doxygenfunction:: specfem::experimental::layout::permute

.. doxygenfunction:: specfem::experimental::layout::group

.. doxygenfunction:: specfem::experimental::layout::ungroup

Selecting subviews
------------------

Include ``specfem/experimental/layout/subview.hpp`` for ``subview``. It accepts
a mapping or an experimental ``View`` and takes exactly one selector per
top-level mode, as ``Kokkos::subview`` does. Selectors mirror coordinates:

* an integer or ``c<N>`` fixes the node's scalar index (first leaf fastest) and
  removes the node;
* ``all`` keeps the whole node with its static types;
* ``range(b, e)`` keeps the half-open scalar range ``[b, e)`` and the node's
  topology;
* a tuple of selectors, one per child, selects inside a nested node. The tuple
  keeps its level even if it loses all but one child, or all children.
  ``all`` and ``range`` are ``TupleElement`` types, so ``make_tuple``
  builds selector tuples directly.

The result topology depends only on selector types. Fixing every top-level mode
gives a rank-zero mapping. Selector positions can be runtime values, including
inside kernels.

.. code-block:: cpp

    #include "specfem/experimental/layout/subview.hpp"

    namespace hl = specfem::experimental::layout;
    // 7 x 3 packed, column-major.
    const auto m = hl::make_layout(hl::make_tuple(7, 3), hl::make_tuple(1, 7));
    const auto column = hl::subview(m, hl::all, 1);       // rank 1, displacement 7
    const auto rows = hl::subview(m, hl::range(2, 7), 1); // rank 1, displacement 9
    // rows.offset(hl::make_tuple(0)) == 9, the parent's (2, 1).

A range on a leaf is always representable. A range on a nested node keeps every
leaf except the slowest one, L, and shortens L. Let P be the product of the
other leaves of that node. A nonempty range must start at a multiple of P. L's
extent becomes ``ceil((e - b) / P)`` and the displacement grows by
``(b / P) * stride(L)``. An unaligned start cannot be expressed with shapes and
strides. Select the parts with a tuple selector instead, for example
``hl::make_tuple(hl::range(b, e), tile)`` for a range inside one tile.

An unaligned end is accepted. The envelope rounds up to whole blocks of P, and
``valid`` rejects scalar indices at or past ``e - b``. The rounded envelope lies
inside the parent's envelope, so offsets stay distinct and inside the parent's
span.

.. code-block:: cpp

    // One mode of shape (4, 3) with strides (1, 10), and a mode of extent 2.
    const auto parent = hl::make_layout(
        hl::make_tuple(hl::make_tuple(4, 3), 2),
        hl::make_tuple(hl::make_tuple(1, 10), 5));
    const auto s = hl::subview(parent, hl::range(4, 10), hl::all);
    // Shape ((4, 2), 2), displacement 10, envelope 16.
    // Scalar 5 has offset 21, as parent scalar 9 does. Scalars 6 and 7 are
    // in the envelope but s.valid(hl::make_tuple(6, 0)) is false.
    const auto lanes = hl::subview(parent, hl::make_tuple(hl::range(2, 4), 1),
                                   hl::all);
    // Lanes 2 and 3 of block 1: shape ((2), 2), displacement 12.

Every valid subview coordinate corresponds to exactly one valid parent
coordinate with the same offset. ``valid`` checks the subview's envelope, then
the range restrictions, then the parent's own ``valid`` at the corresponding
coordinate. Holes, disabled domains and earlier range restrictions of the
parent therefore carry over, including through ``permute``, ``group`` and
``ungroup`` in either order. ``offset`` adds no work beyond the subview's own
descriptor in release builds.

``all`` and ``range(hl::c<0>, n)`` add a static zero, so selecting with them
alone keeps the parent's displacement type. A fixed index or a nonzero range
start adds a compile-time constant only when the index or both range bounds are
constants and the node's shape and strides are fully static. Otherwise the
displacement is a runtime ``std::size_t``. A subview of an empty or disabled
parent computes no displacement terms at runtime, because such a parent need not
have a representable span.

An empty range (``b == e``) at any position gives an empty subview. It skips
the alignment check and the displacement, and the selected extent is zero.
``empty()`` is conservative for subviews: true means no coordinate is valid,
but false does not prove that one is.

A view result retains the parent's backing handle, capacity and element
constness. Managed views keep their allocation alive after the parents leave
scope. Borrowed storage follows the rules of the parent view. ``subview``
allocates nothing and copies no elements.

Selector counts and tuple arities that do not match the topology, tuple
selectors on leaves, unsupported selector types, and invalid constant indices
or ranges on static extents fail compilation in every build mode. Runtime
checks in builds without ``NDEBUG`` report, in this order, "layout subview
index out of range", "layout subview range out of order or out of range", and
"layout subview range start not aligned to nested mode blocks", before any
division. Displacement arithmetic is checked for overflow. Accessing a coordinate
that ``valid`` rejects reports "layout subview coordinate out of bounds". In
release builds these checks are absent and invalid requests are undefined
behavior. Stepped, reversed and broadcast selections are not supported.

.. doxygenstruct:: specfem::experimental::layout::All

.. doxygenvariable:: specfem::experimental::layout::all

.. doxygenstruct:: specfem::experimental::layout::Range
    :members:

.. doxygenfunction:: specfem::experimental::layout::range

.. doxygenfunction:: specfem::experimental::layout::subview
