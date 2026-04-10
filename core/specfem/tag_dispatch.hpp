#pragma once

/**
 * @defgroup tag_dispatch_group specfem::tag_dispatch
 * @{
 */

/**
 * @brief Compile-time element-type dispatch infrastructure.
 *
 * `tag_dispatch` provides a composable, zero-overhead mechanism for
 * enumerating every **valid** combination of element tags
 * (dimension × medium × property × attenuation × boundary), iterating over
 * those combinations at compile time, and storing per-combination data in
 * type-safe containers.
 */
namespace specfem::tag_dispatch {}

/**
 * @section Motivation
 *
 * SPECFEM++ supports many element flavours — `elastic_psv`, `elastic_sh`,
 * `acoustic`, etc. in 2-D; `elastic`, `acoustic` in 3-D — each with its own
 * material property, attenuation, and boundary-condition variants.  The raw
 * Cartesian product of all tag enumerators contains many physically
 * meaningless entries (e.g. dim3 + elastic_sh).  `tag_dispatch` encodes the
 * validity rules once (in `is_valid.hpp`) and propagates them automatically to
 * every higher-level facility.
 *
 * @section Step 1 — Declare an element set with `element_combinations`
 *
 * Compose named tag-set types using `operator*` to describe which tag values
 * are in play.  The resulting type, `ET`, holds a `constexpr` array
 * `ET::combos` of only the **valid** tuples:
 *
 * @code
 * namespace td  = specfem::tag_dispatch;
 * namespace el  = specfem::element;
 *
 * using ET = decltype(
 *     td::dimension_set <el::dimension_tag::dim2>{} *
 *     td::medium_set    <el::medium_tag::elastic_psv,
 *                        el::medium_tag::elastic_sh,
 *                        el::medium_tag::acoustic>{} *
 *     td::property_set  <el::property_tag::isotropic>{} *
 *     td::attenuation_set<el::attenuation_tag::none,
 *                         el::attenuation_tag::constant_isotropic>{} *
 *     td::boundary_set  <el::boundary_tag::none,
 *                        el::boundary_tag::stacey>{});
 *
 * // ET::size   = number of valid combos (invalid cross-products are dropped)
 * // ET::combos = constexpr array of TagValueTuples, e.g.:
 * //   combos[0] = (dim2, elastic_psv, isotropic, none,               none)
 * //   combos[1] = (dim2, elastic_psv, isotropic, none,               stacey)
 * //   combos[2] = (dim2, elastic_psv, isotropic, constant_isotropic, none)
 * //   combos[3] = (dim2, elastic_psv, isotropic, constant_isotropic, stacey)
 * //   combos[4] = (dim2, elastic_sh,  isotropic, none,               none)
 * //   ...  (acoustic combos follow; dim2+elastic_sh+stacey omitted as invalid)
 * @endcode
 *
 * Each entry in `ET::combos` corresponds to one concrete element flavour.
 * The combo at index `I` maps directly to a `specfem::tags::Tags<...>` type
 * via `combo_to_tags_t<ET, I, ET::combo_type::arity>`.
 *
 * @section Step 2 — Iterate with `for_each`
 *
 * `for_each` calls a generic lambda once per valid combo, passing the
 * combo's `Tags<...>` type as a template argument.  This is the primary
 * mechanism for instantiating type-specialised kernels or populating
 * metadata at startup:
 *
 * @code
 * td::for_each(ET{}, []<typename TagsType>() {
 *     // TagsType == specfem::tags::Tags<dim2, elastic_psv, isotropic, none,
 * none>
 *     // for the first combo, then Tags<...> for each subsequent valid combo.
 *     setup_kernel<TagsType>();
 * });
 * @endcode
 *
 * @section Step 3 — Store per-combo data with `Storage` or `TypedStorage`
 *
 * **`Storage<T, ET>`** — every combo holds the same type `T`.  Suitable for
 * Kokkos views, counters, or any homogeneous per-combo resource:
 *
 * @code
 * td::Storage<Kokkos::View<double*>, ET> fields(
 *     []<typename TagsType>() {
 *         return Kokkos::View<double*>("displacement", n_elems);
 *     });
 *
 * // Compile-time access — zero-overhead base-class upcast:
 * using T = specfem::tags::Tags<
 *     el::dimension_tag::dim2,
 *     el::medium_tag::elastic_psv,
 *     el::property_tag::isotropic,
 *     el::attenuation_tag::none,
 *     el::boundary_tag::none>;
 * Kokkos::View<double*>& v = fields.get<T>();
 *
 * // Runtime access — linear scan, host only, homogeneous stores only:
 * auto& v2 = fields.get(el::medium_tag::elastic_psv,
 *                       el::boundary_tag::none);
 * @endcode
 *
 * **`TypedStorage<Tmpl, ET>`** — each combo holds `Tmpl<TagsType>`, giving a
 * different concrete type per slot.  Suitable for type-specialised kernel
 * objects or policy structs:
 *
 * @code
 * template <typename TagsType> struct Assembler { void run(); };
 *
 * td::TypedStorage<Assembler, ET> assemblers(
 *     []<typename TagsType>() { return Assembler<TagsType>{}; });
 *
 * Assembler<T>& a = assemblers.get<T>();  // T as defined above
 * a.run();
 * @endcode
 *
 * Both container types support `deep_copy(dest, src)` for Kokkos host/device
 * transfers (requires the same `ET` but allows different policies, e.g. device
 * view vs host-mirror view).
 *
 * @section Putting it all together
 *
 * A typical assembly pattern declares the element set once, builds storage
 * from it, and then drives all per-combo work through `for_each`:
 *
 * @code
 * // 1. Element set (usually a shared type alias)
 * using ET = decltype(td::dimension_set<el::dimension_tag::dim2>{} *
 *                     td::medium_set<el::medium_tag::elastic_psv,
 *                                    el::medium_tag::acoustic>{} *
 *                     td::property_set<el::property_tag::isotropic>{} *
 *                     td::attenuation_set<el::attenuation_tag::none>{} *
 *                     td::boundary_set<el::boundary_tag::none,
 *                                      el::boundary_tag::stacey>{});
 *
 * // 2. Storage
 * td::Storage<Kokkos::View<int*>, ET> element_counts(
 *     []<typename TagsType>() { return Kokkos::View<int*>("counts", 1); });
 *
 * td::TypedStorage<Assembler, ET> assemblers(
 *     []<typename TagsType>() { return Assembler<TagsType>{}; });
 *
 * // 3. Dispatch
 * td::for_each(ET{}, [&]<typename TagsType>() {
 *     auto& counts = element_counts.get<TagsType>();
 *     auto& asm    = assemblers.get<TagsType>();
 *     asm.assemble(counts);
 * });
 * @endcode
 *
 * Component summary
 * -----------------
 *
 * | Header                                  | Provides |
 * |-----------------------------------------|-------------------------------------------------------|
 * | `tag_dispatch/is_valid.hpp`             | `TagValueTuple`,
 * `is_valid_*_combo` predicates        | |
 * `tag_dispatch/element_combinations.hpp` | Named tag-set types,
 * `element_combinations`           | | `tag_dispatch/for_each.hpp` |
 * `for_each(ET, lambda)` compile-time iteration         | |
 * `tag_dispatch/find_in.hpp`              | `find_in_t` / `find_in_v`
 * compile-time index lookup   | | `tag_dispatch/storage.hpp`              |
 * `Storage<T,ET>`, `TypedStorage<Tmpl,ET>`, `deep_copy` |
 *
 * @} end of tag_dispatch_group
 */

#include "specfem/macros.hpp"
#include "tag_dispatch/element_combinations.hpp"
#include "tag_dispatch/for_each.hpp"
#include "tag_dispatch/storage.hpp"
