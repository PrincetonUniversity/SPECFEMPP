#pragma once

#include "specfem/assembly/assembly.hpp"
#include "specfem/datatype/element_index_range.hpp"
#include "specfem/element.hpp"
#include "specfem/enums.hpp"
#include "specfem/linear_system/sparse_matrix_view/dof_set.hpp"
#include "specfem/tags.hpp"
#include <Kokkos_Core.hpp>
#include <concepts>
#include <cstddef>
#include <type_traits>
#include <vector>

namespace specfem {
namespace linear_system {

/**
 * @brief Maps a SPECFEM++ mesh point of one medium to a solver degree of
 * freedom.
 *
 * Answers "given `(ispec, iz, iy, ix, icomp)`, which global dof is that?", and
 * carries the connectivity a sparse matrix over those dofs is built from: the
 * element range of the medium, the dof ids of each element
 * (@ref element_dofs), and the mesh points an absorbing boundary acts at
 * (@ref is_damping_point).
 *
 * The global dof id is the offset a `Kokkos::layout_left` mapping over the
 * extents `(nglob, ncomponents)` gives to `(iglob, icomp)`, which is exactly
 * how SPECFEM++ stores a field (`Kokkos::View<type_real **,
 * Kokkos::LayoutLeft>` of the same shape). A solver vector therefore maps 1:1
 * onto field memory with no permutation at solve time. Holding the layout
 * mapping rather than open-coding the offset means the ordering is stated once,
 * by the layout type: swap @ref component_layout_type for
 * `Kokkos::layout_right` to block by point instead of by component.
 *
 * One instance describes one medium: `nglob` and `ncomp` are per-medium
 * quantities (a future multi-medium system holds one mapping per medium).
 *
 * This class holds SPECFEM++ quantities only and names no linear-algebra
 * library type, so it compiles in a build without Trilinos and can be reused
 * unchanged if the solver backend is swapped. The maps and sparsity graphs
 * built from it live in @ref FEAssembly.
 *
 * @tparam DimensionTag Spatial dimension; must be `dim3`
 * @tparam MediumTag Medium whose degrees of freedom are numbered
 * @tparam GlobalOrdinal Integer type of the global dof ids; supplied by the
 *                       linear-algebra backend (see
 *                       specfem::linear_system::FEMapping for the Tpetra
 *                       binding)
 */
template <specfem::element::dimension_tag DimensionTag,
          specfem::element::medium_tag MediumTag, typename GlobalOrdinal>
  requires(DimensionTag == specfem::element::dimension_tag::dim3)
class Mapping {
public:
  /// Integer type of the global dof ids
  using global_ordinal_type = GlobalOrdinal;

  /// Assembly this mapping is built from
  using AssemblyType = specfem::assembly::assembly<DimensionTag>;

  constexpr static auto dimension_tag = DimensionTag;
  constexpr static auto medium_tag = MediumTag;

  /// Field components per mesh point of the medium (3 for 3D elastic)
  constexpr static int ncomponents =
      specfem::element::attributes<DimensionTag, MediumTag>::components;

  /**
   * @brief `(ispec, iz, iy, ix)` to global point index, on the host.
   *
   * The host mirror of the simulation field's `index_mapping`: connectivity is
   * walked in host loops, so the device view itself is not addressable here.
   * Spelled out rather than named through the field because the field's own
   * `IndexViewType` alias is private.
   */
  using index_view_type =
      Kokkos::View<int ****, Kokkos::LayoutLeft,
                   Kokkos::DefaultExecutionSpace>::host_mirror_type;

  /// Half-open index range of the medium's elements; `View`-compatible
  using element_range_type = specfem::datatype::ElementIndexRange;

  /// Extents of the dof grid: `(nglob, ncomponents)`
  using extents_type = Kokkos::dextents<std::size_t, 2>;

  /// Layout that turns `(iglob, icomp)` into a global dof id
  using component_layout_type = Kokkos::layout_left::mapping<extents_type>;

  /**
   * @brief Number the degrees of freedom of one medium of an assembled
   * simulation.
   *
   * Reads the index mapping, the per-medium point count and dof base offset
   * from the forward simulation field, the element range and grid from the
   * element types, and the absorbing-boundary mask from the boundaries.
   *
   * The assembly is not retained: everything the mapping needs is copied or
   * view-shared at construction.
   *
   * @param assembly Assembly with constructed fields
   */
  explicit Mapping(const AssemblyType &assembly)
      : elements_(assembly.element_types.get_elements_on_host(medium_tag)),
        ngllz_(assembly.element_types.element_grid.ngllz),
        nglly_(assembly.element_types.element_grid.nglly),
        ngllx_(assembly.element_types.element_grid.ngllx),
        nglob_(forward_field(assembly).template get_nglob<medium_tag>()),
        dof_base_(
            forward_field(assembly)
                .dof_ranges
                .template get<specfem::tags::Tags<DimensionTag, MediumTag>>()
                .begin_index()),
        mapping_(forward_field(assembly).h_index_mapping),
        component_layout_(extents_type(nglob_, ncomponents)),
        damping_mask_(build_damping_mask(assembly)) {}

  /**
   * @brief Global dof id of component `icomp` at mesh point `iglob`.
   *
   * The component-blocked offset of `(iglob, icomp)` under
   * @ref component_layout_type -- the single source of truth for the global
   * dof ordering (see the class docs).
   *
   * @param iglob Per-medium global point index in `[0, nglob())`
   * @param icomp Field component in `[0, ncomp())`
   * @return Global dof id in `[0, num_global_dofs())`
   */
  inline global_ordinal_type operator()(const int iglob,
                                        const int icomp) const {
    return static_cast<global_ordinal_type>(component_layout_(iglob, icomp));
  }

  /**
   * @brief Global dof id of component `icomp` at a GLL point of an element.
   *
   * @param ispec Element index
   * @param iz Quadrature index in z
   * @param iy Quadrature index in y
   * @param ix Quadrature index in x
   * @param icomp Field component in `[0, ncomp())`
   * @return Global dof id in `[0, num_global_dofs())`
   */
  inline global_ordinal_type operator()(const int ispec, const int iz,
                                        const int iy, const int ix,
                                        const int icomp) const {
    return (*this)(iglob(ispec, iz, iy, ix), icomp);
  }

  /**
   * @brief Name a *set* of dofs by GLL coordinate.
   *
   * The five-slot selector form of @ref operator(): each slot is an integral,
   * `Kokkos::ALL`, or a container of indices (see @ref IndexSelector), and the
   * result is a lazily-expanded @ref DofSet rather than a single id. The
   * all-integral case is handled by the scalar overload above.
   *
   * `Kokkos::ALL` in the element slot means every element *of this medium*, so
   * it is normalized here to @ref elements rather than to `[0, nspec)` -- the
   * mapping is per-medium and a mesh may hold elements of others.
   *
   * @code
   * // the 375 dofs of one element, in local_dof_index order
   * mapping(ispec, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
   * // the dofs of a batch of elements, element-major
   * mapping(batch, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
   * @endcode
   *
   * @tparam ISpec Element slot selector
   * @tparam IZ Quadrature-z slot selector
   * @tparam IY Quadrature-y slot selector
   * @tparam IX Quadrature-x slot selector
   * @tparam IComp Component slot selector
   * @param ispec Element(s)
   * @param iz Quadrature index/indices in z
   * @param iy Quadrature index/indices in y
   * @param ix Quadrature index/indices in x
   * @param icomp Field component(s)
   * @return Dof set; borrows this mapping and must not outlive it
   */
  template <typename ISpec, typename IZ, typename IY, typename IX,
            typename IComp>
    requires(IndexSelector<ISpec> && IndexSelector<IZ> && IndexSelector<IY> &&
             IndexSelector<IX> && IndexSelector<IComp> &&
             !(std::integral<ISpec> && std::integral<IZ> && std::integral<IY> &&
               std::integral<IX> && std::integral<IComp>))
  auto operator()(ISpec ispec, IZ iz, IY iy, IX ix, IComp icomp) const {
    auto elements = normalize_element_selector(ispec);
    return DofSet<Mapping, decltype(elements), IZ, IY, IX, IComp>(
        *this, elements, iz, iy, ix, icomp);
  }

  /**
   * @brief Name a *set* of dofs by mesh point.
   *
   * The two-slot selector form of @ref operator(); see the five-slot overload
   * above. `Kokkos::ALL` in the point slot means every point of the medium,
   * `[0, nglob())`.
   *
   * @code
   * // the ncomp dofs at one mesh point -- a damping block's rows
   * mapping(iglob, Kokkos::ALL);
   * @endcode
   *
   * @tparam IGlob Point slot selector
   * @tparam IComp Component slot selector
   * @param iglob Mesh point(s)
   * @param icomp Field component(s)
   * @return Dof set; borrows this mapping and must not outlive it
   */
  template <typename IGlob, typename IComp>
    requires(IndexSelector<IGlob> && IndexSelector<IComp> &&
             !(std::integral<IGlob> && std::integral<IComp>))
  auto operator()(IGlob iglob, IComp icomp) const {
    return DofSet<Mapping, IGlob, IComp>(*this, iglob, icomp);
  }

  /**
   * @brief Global dof ids of one element, in element-local dof order.
   *
   * Entry `ldof` holds the id of the dof that @ref local_dof_index numbers
   * `ldof`, so the same vector orders both a sparsity-graph insert and a dense
   * element-block scatter.
   *
   * @param ispec Element index
   * @return `ncomp * ngllz * nglly * ngllx` global dof ids
   */
  std::vector<global_ordinal_type> element_dofs(const int ispec) const {
    const int npoints = ngllz_ * nglly_ * ngllx_;
    std::vector<global_ordinal_type> dofs(
        static_cast<std::size_t>(ncomponents) * npoints);

    for (int iz = 0; iz < ngllz_; ++iz) {
      for (int iy = 0; iy < nglly_; ++iy) {
        for (int ix = 0; ix < ngllx_; ++ix) {
          const int point = (iz * nglly_ + iy) * ngllx_ + ix;
          const int point_iglob = iglob(ispec, iz, iy, ix);
          for (int icomp = 0; icomp < ncomponents; ++icomp) {
            dofs[static_cast<std::size_t>(icomp) * npoints + point] =
                (*this)(point_iglob, icomp);
          }
        }
      }
    }
    return dofs;
  }

  /**
   * @brief Whether an absorbing boundary traction acts at a mesh point.
   *
   * True where any GLL point mapping to `iglob` carries
   * specfem::element::boundary_tag::stacey or
   * specfem::element::boundary_tag::composite_stacey_dirichlet.
   *
   * Shared with whatever fills a damping matrix over this numbering: a scatter
   * that disagrees with this mask would update fewer entries than the block
   * size and leave the matrix inconsistent with its own sparsity.
   *
   * @param iglob Per-medium global point index in `[0, nglob())`
   * @return `true` if the point contributes a damping block
   */
  inline bool is_damping_point(const int iglob) const {
    return damping_mask_[iglob];
  }

  /// Number of unique global mesh points of the medium
  inline int nglob() const { return nglob_; }

  /// Number of field components per mesh point
  inline int ncomp() const { return ncomponents; }

  /// Number of elements of the medium
  inline int nelements() const { return elements_.size(); }

  /// Half-open index range of the medium's elements
  inline const element_range_type &elements() const { return elements_; }

  /// Number of quadrature points in z
  inline int ngllz() const { return ngllz_; }

  /// Number of quadrature points in y
  inline int nglly() const { return nglly_; }

  /// Number of quadrature points in x
  inline int ngllx() const { return ngllx_; }

  /// Total number of degrees of freedom: `ncomp * nglob`
  inline global_ordinal_type num_global_dofs() const {
    return static_cast<global_ordinal_type>(ncomponents) * nglob_;
  }

private:
  /// Forward simulation field of `assembly`
  static const auto &forward_field(const AssemblyType &assembly) {
    return assembly.fields.template get_simulation_field<
        specfem::simulation::field_type::forward>();
  }

  /**
   * @brief Per-medium global point index of a GLL point.
   *
   * `index_mapping` numbers points across all media; subtracting the medium's
   * `dof_ranges` base compacts that to `[0, nglob())`.
   */
  inline int iglob(const int ispec, const int iz, const int iy,
                   const int ix) const {
    return mapping_(ispec, iz, iy, ix) - dof_base_;
  }

  /**
   * @brief Resolve `Kokkos::ALL` in the element slot to this medium's elements.
   *
   * Every other slot indexes a dense range, so `Kokkos::ALL` there is the
   * identity. The element slot is the exception: a mesh may hold elements of
   * other media, so "all elements" means @ref elements, not `[0, nspec)`.
   * Doing the substitution here keeps @ref DofSet free of any slot-specific
   * special case.
   *
   * @tparam Selector Element slot selector
   * @param selector Element slot value
   * @return @ref elements if `selector` is `Kokkos::ALL`, otherwise `selector`
   */
  template <typename Selector>
  auto normalize_element_selector(Selector selector) const {
    if constexpr (std::is_same_v<std::remove_cvref_t<Selector>,
                                 Kokkos::ALL_t>) {
      return elements_;
    } else {
      return selector;
    }
  }

  /**
   * @brief Mark the mesh points an absorbing boundary traction acts at.
   *
   * A construction-time extractor: the boundary data is read here and not
   * retained, so the mapping holds the mask rather than the boundary views.
   *
   * A mesh point is shared by several elements, so the mask is accumulated
   * over every element touching it and is indexed by `iglob` rather than by
   * `(ispec, iz, iy, ix)`.
   *
   * @param assembly Assembly whose boundaries are queried
   * @return Mask of length `nglob()`
   */
  std::vector<bool> build_damping_mask(const AssemblyType &assembly) const {
    std::vector<bool> mask(nglob_, false);

    for (int i = 0; i < elements_.size(); ++i) {
      const int ispec = elements_(i);
      for (int iz = 0; iz < ngllz_; ++iz) {
        for (int iy = 0; iy < nglly_; ++iy) {
          for (int ix = 0; ix < ngllx_; ++ix) {
            const auto tag =
                assembly.boundaries.get_boundary_tag_on_host(ispec, iz, iy, ix);
            if (tag == specfem::element::boundary_tag::stacey ||
                tag == specfem::element::boundary_tag::
                           composite_stacey_dirichlet) {
              mask[iglob(ispec, iz, iy, ix)] = true;
            }
          }
        }
      }
    }
    return mask;
  }

  element_range_type elements_; ///< Elements of the medium

  int ngllz_;    ///< Quadrature points in z
  int nglly_;    ///< Quadrature points in y
  int ngllx_;    ///< Quadrature points in x
  int nglob_;    ///< Points of the medium
  int dof_base_; ///< All-media base offset of this medium's points

  index_view_type mapping_; ///< (ispec, iz, iy, ix) -> all-media point index
  component_layout_type component_layout_; ///< (iglob, icomp) -> global dof id

  // Declared after everything build_damping_mask reads, so that the
  // member-initializer list fills it from complete state.
  std::vector<bool> damping_mask_; ///< Absorbing-boundary points, by iglob
};

} // namespace linear_system
} // namespace specfem
