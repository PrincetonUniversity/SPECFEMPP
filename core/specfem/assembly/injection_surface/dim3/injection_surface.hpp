#pragma once

#include "specfem/assembly/boundaries.hpp"
#include "specfem/assembly/element_types.hpp"
#include "specfem/assembly/impl/value_containers.hpp"
#include "specfem/assembly/jacobian_matrix.hpp"
#include "specfem/assembly/mesh.hpp"
#include "specfem/datatype/element_index_range.hpp"
#include "specfem/enums.hpp"
#include "specfem/mesh.hpp"
#include "specfem/tag_dispatch.hpp"
#include <Kokkos_Core.hpp>

namespace specfem::assembly {

/**
 * @brief Quadrature points of the wavefield-injection surface (3D)
 *
 * One @e entry per (element, injection face, GLL point on that face). A GLL
 * point on two injection faces of the same element gives two entries, each
 * with its own face normal and weight. Every entry maps to a @e row, a point
 * handed to the injection plugin; in this version each entry has its own row.
 * Kernels read the plugin's values for an entry through row().
 *
 * Entries and rows are ordered by material system, in the order of
 * specfem::assembly::impl::ContainerSets<dim3>::combinations, so the entries
 * and rows of one (medium, property) pair are contiguous ranges.
 *
 * Normals are unit vectors pointing out of the total-field region (out of the
 * domain on the box sides of the mesh), and weights are the surface
 * quadrature weights dS, so the surface integral of f is the sum over entries
 * of weight * f.
 */
template <> struct InjectionSurface<specfem::element::dimension_tag::dim3> {
  constexpr static auto dimension_tag =
      specfem::element::dimension_tag::dim3; ///< Dimension tag

  /// Valid (medium, property) systems, in group order
  constexpr static auto combinations =
      specfem::assembly::impl::ContainerSets<dimension_tag>::combinations;

  using IndexView = Kokkos::View<int *, Kokkos::DefaultExecutionSpace>;
  using NormalView =
      Kokkos::View<type_real *[3], Kokkos::DefaultExecutionSpace>;
  using WeightView = Kokkos::View<type_real *, Kokkos::DefaultExecutionSpace>;
  using BoundaryTagView = Kokkos::View<specfem::element::boundary_tag *,
                                       Kokkos::DefaultExecutionSpace>;
  /// Per-row host arrays; the same type as
  /// specfem::injection::InjectionPoints::view_type, so they can be shared.
  using RowView = Kokkos::View<type_real *, Kokkos::HostSpace>;
  using RangeStorage =
      specfem::tag_dispatch::Storage<specfem::datatype::ElementIndexRange,
                                     decltype(combinations)>;

  int nentries = 0; ///< Number of entries on this rank
  int nrows = 0;    ///< Number of rows on this rank

  IndexView ispec;                       ///< Compute element of each entry
  IndexView::host_mirror_type h_ispec;   ///< Host mirror
  IndexView iz;                          ///< GLL index in z of each entry
  IndexView::host_mirror_type h_iz;      ///< Host mirror
  IndexView iy;                          ///< GLL index in y of each entry
  IndexView::host_mirror_type h_iy;      ///< Host mirror
  IndexView ix;                          ///< GLL index in x of each entry
  IndexView::host_mirror_type h_ix;      ///< Host mirror
  IndexView row;                         ///< Plugin row of each entry
  IndexView::host_mirror_type h_row;     ///< Host mirror
  NormalView normal;                     ///< Unit normal of each entry
  NormalView::host_mirror_type h_normal; ///< Host mirror
  WeightView weight;                     ///< Surface weight dS of each entry
  WeightView::host_mirror_type h_weight; ///< Host mirror
  BoundaryTagView boundary_tag; ///< Boundary tag of each entry's GLL point
  BoundaryTagView::host_mirror_type h_boundary_tag; ///< Host mirror

  RowView x;  ///< x-coordinate of each row
  RowView y;  ///< y-coordinate of each row
  RowView z;  ///< z-coordinate of each row
  RowView nx; ///< Unit normal of each row, x-component
  RowView ny; ///< Unit normal of each row, y-component
  RowView nz; ///< Unit normal of each row, z-component
  Kokkos::View<int *, Kokkos::HostSpace> h_row_entry; ///< An entry of each
                                                      ///< row

  RangeStorage entry_ranges; ///< Entries of each material system
  RangeStorage row_ranges;   ///< Rows of each material system

  ///@{
  InjectionSurface() = default;

  /**
   * @brief Build the entries from the injection faces declared in the mesh
   *
   * @tparam ModelTag Mesh model
   * @param mesh Mesh, whose boundaries.injection_surface lists the faces
   * @param mesh_assembly Assembled mesh (coordinates, GLL weights, element
   * mapping)
   * @param jacobian_matrix Jacobian matrix, for face normals and weights
   * @param element_types Medium and property of every element
   * @param boundaries Boundary conditions, for the tag of each GLL point
   * @throws std::runtime_error if a face is not a whole element face or has a
   * degenerate normal
   */
  template <specfem::simulation::model ModelTag>
  InjectionSurface(
      const specfem::mesh::mesh<ModelTag> &mesh,
      const specfem::assembly::mesh<dimension_tag> &mesh_assembly,
      const specfem::assembly::jacobian_matrix<dimension_tag> &jacobian_matrix,
      const specfem::assembly::element_types<dimension_tag> &element_types,
      const specfem::assembly::boundaries<dimension_tag> &boundaries);
  ///@}

  /**
   * @brief Entries of one material system
   * @tparam MediumTag Medium
   * @tparam PropertyTag Property
   * @return Contiguous range of entry indices
   */
  template <specfem::element::medium_tag MediumTag,
            specfem::element::property_tag PropertyTag>
  KOKKOS_INLINE_FUNCTION specfem::datatype::ElementIndexRange
  get_entries() const {
    return entry_ranges.template get<
        specfem::tags::Tags<dimension_tag, MediumTag, PropertyTag>>();
  }

  /**
   * @brief Rows of one material system
   * @tparam MediumTag Medium
   * @tparam PropertyTag Property
   * @return Contiguous range of row indices
   */
  template <specfem::element::medium_tag MediumTag,
            specfem::element::property_tag PropertyTag>
  KOKKOS_INLINE_FUNCTION specfem::datatype::ElementIndexRange get_rows() const {
    return row_ranges.template get<
        specfem::tags::Tags<dimension_tag, MediumTag, PropertyTag>>();
  }
};

} // namespace specfem::assembly
