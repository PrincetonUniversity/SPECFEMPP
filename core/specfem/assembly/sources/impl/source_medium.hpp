#pragma once

#include "specfem/assembly/compute_source_array.hpp"
#include "specfem/assembly/element_types.hpp"
#include "specfem/assembly/jacobian_matrix.hpp"
#include "specfem/assembly/mesh.hpp"
#include "specfem/point.hpp"
#include "specfem/setup.hpp"
#include "specfem/source.hpp"
#include <Kokkos_Core.hpp>
#include <memory>
#include <tuple>
#include <type_traits>
#include <vector>

// Forward declarations
namespace specfem {
namespace algorithms {
template <specfem::element::dimension_tag DimensionTag>
specfem::point::local_coordinates<DimensionTag> locate_point(
    const specfem::point::global_coordinates<DimensionTag> &coordinates,
    const specfem::assembly::mesh<DimensionTag> &mesh);
}
} // namespace specfem

namespace specfem::assembly::sources_impl {

template <typename T, int Rank> struct ExtentImpl {
  using type = typename ExtentImpl<T, Rank - 1>::type *;
};
template <typename T> struct ExtentImpl<T, 0> {
  using type = T;
};

/**
 * @brief Medium-specific source data management for spectral element
 * simulations
 *
 * Manages source time functions, Lagrange interpolants, and element mappings
 * for a specific (dimension, medium) pair.
 *
 * @tparam DimensionTag Spatial dimension (`dim2` or `dim3`)
 * @tparam MediumTag Physical medium type (`elastic_psv`, `acoustic`, etc.)
 */
template <specfem::element::dimension_tag DimensionTag,
          specfem::element::medium_tag MediumTag>
struct source_medium {

  constexpr static auto medium_tag = MediumTag;
  constexpr static auto dimension_tag = DimensionTag;
  constexpr static auto ndim =
      (DimensionTag == specfem::element::dimension_tag::dim2) ? 2 : 3;
  constexpr static int source_array_rank = ndim + 2;

private:
  using IndexView = Kokkos::View<int *, Kokkos::DefaultExecutionSpace>;
  using SourceTimeFunctionView =
      Kokkos::View<type_real ***, Kokkos::LayoutRight,
                   Kokkos::DefaultExecutionSpace>;
  using SourceArrayView =
      Kokkos::View<typename specfem::assembly::sources_impl::ExtentImpl<
                       type_real, source_array_rank>::type,
                   Kokkos::LayoutRight, Kokkos::DefaultExecutionSpace>;

  constexpr static int components =
      specfem::element::attributes<dimension_tag, MediumTag>::components;

public:
  source_medium() = default;

  // ── Constructor (dim2) ───────────────────────────────────────────────────
  template <specfem::element::dimension_tag U = DimensionTag,
            typename std::enable_if<
                U == specfem::element::dimension_tag::dim2>::type * = nullptr>
  source_medium(
      const std::vector<
          std::shared_ptr<specfem::sources::source<DimensionTag> > > &sources,
      const specfem::assembly::mesh<DimensionTag> &mesh,
      const specfem::assembly::jacobian_matrix<DimensionTag> &jacobian_matrix,
      const specfem::assembly::element_types<DimensionTag> &element_types,
      const type_real t0, const type_real dt, const int nsteps)
      : source_index_mapping("specfem::sources::source_index_mapping",
                             sources.size()),
        h_source_index_mapping(
            Kokkos::create_mirror_view(source_index_mapping)),
        source_time_function("specfem::sources::source_time_function", nsteps,
                             sources.size(), components),
        h_source_time_function(
            Kokkos::create_mirror_view(source_time_function)),
        source_array("specfem::sources::source_array", sources.size(),
                     components, mesh.element_grid.ngllz,
                     mesh.element_grid.ngllx),
        h_source_array(Kokkos::create_mirror_view(source_array)) {

    for (int isource = 0; isource < (int)sources.size(); isource++) {
      auto sv_source_array = Kokkos::subview(
          this->h_source_array, isource, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
      specfem::assembly::compute_source_array(sources[isource], mesh,
                                              jacobian_matrix, sv_source_array);
      auto sv_stf_array = Kokkos::subview(this->h_source_time_function,
                                          Kokkos::ALL, isource, Kokkos::ALL);
      sources[isource]->compute_source_time_function(t0, dt, nsteps,
                                                     sv_stf_array);
      this->h_source_index_mapping(isource) =
          sources[isource]->get_local_coordinates().ispec;
    }

    Kokkos::deep_copy(source_array, h_source_array);
    Kokkos::deep_copy(source_time_function, h_source_time_function);
    Kokkos::deep_copy(source_index_mapping, h_source_index_mapping);
  }

  // ── Constructor (dim3) ───────────────────────────────────────────────────
  template <specfem::element::dimension_tag U = DimensionTag,
            typename std::enable_if<
                U == specfem::element::dimension_tag::dim3>::type * = nullptr>
  source_medium(
      const std::vector<
          std::shared_ptr<specfem::sources::source<DimensionTag> > > &sources,
      const specfem::assembly::mesh<DimensionTag> &mesh,
      const specfem::assembly::jacobian_matrix<DimensionTag> &jacobian_matrix,
      const specfem::assembly::element_types<DimensionTag> &element_types,
      const type_real t0, const type_real dt, const int nsteps)
      : source_index_mapping("specfem::sources::source_index_mapping",
                             sources.size()),
        h_source_index_mapping(
            Kokkos::create_mirror_view(source_index_mapping)),
        source_time_function("specfem::sources::source_time_function", nsteps,
                             sources.size(), components),
        h_source_time_function(
            Kokkos::create_mirror_view(source_time_function)),
        source_array("specfem::sources::source_array", sources.size(),
                     components, mesh.element_grid.ngllz,
                     mesh.element_grid.nglly, mesh.element_grid.ngllx),
        h_source_array(Kokkos::create_mirror_view(source_array)) {

    for (int isource = 0; isource < (int)sources.size(); isource++) {
      auto sv_source_array =
          Kokkos::subview(this->h_source_array, isource, Kokkos::ALL,
                          Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
      specfem::assembly::compute_source_array(sources[isource], mesh,
                                              jacobian_matrix, sv_source_array);
      auto sv_stf_array = Kokkos::subview(this->h_source_time_function,
                                          Kokkos::ALL, isource, Kokkos::ALL);
      sources[isource]->compute_source_time_function(t0, dt, nsteps,
                                                     sv_stf_array);
      this->h_source_index_mapping(isource) =
          sources[isource]->get_local_coordinates().ispec;
    }

    Kokkos::deep_copy(source_array, h_source_array);
    Kokkos::deep_copy(source_time_function, h_source_time_function);
    Kokkos::deep_copy(source_index_mapping, h_source_index_mapping);
  }

  // ── Data members ─────────────────────────────────────────────────────────

  IndexView source_index_mapping;
  IndexView::HostMirror h_source_index_mapping;

  SourceTimeFunctionView source_time_function;
  SourceTimeFunctionView::HostMirror h_source_time_function;

  SourceArrayView source_array;
  typename SourceArrayView::HostMirror h_source_array;

  // ── Data access: load_on_device (dim2) ───────────────────────────────────

  template <typename IndexType, typename PointSourceType,
            specfem::element::dimension_tag U = DimensionTag,
            typename std::enable_if<
                U == specfem::element::dimension_tag::dim2>::type * = nullptr>
  KOKKOS_INLINE_FUNCTION void
  load_on_device(const int timestep, const IndexType &index,
                 PointSourceType &point_source) const {
    const auto isource = index.imap;
    for (int component = 0; component < components; component++) {
      point_source.stf(component) =
          source_time_function(timestep, isource, component);
      point_source.lagrange_interpolant(component) =
          source_array(isource, component, index.iz, index.ix);
    }
  }

  // ── Data access: load_on_device (dim3) ───────────────────────────────────

  template <typename IndexType, typename PointSourceType,
            specfem::element::dimension_tag U = DimensionTag,
            typename std::enable_if<
                U == specfem::element::dimension_tag::dim3>::type * = nullptr>
  KOKKOS_INLINE_FUNCTION void
  load_on_device(const int timestep, const IndexType &index,
                 PointSourceType &point_source) const {
    const auto isource = index.imap;
    for (int component = 0; component < components; component++) {
      point_source.stf(component) =
          source_time_function(timestep, isource, component);
      point_source.lagrange_interpolant(component) =
          source_array(isource, component, index.iz, index.iy, index.ix);
    }
  }

  // ── Data access: store_on_device (dim2) ──────────────────────────────────

  template <typename IndexType, typename PointSourceType,
            specfem::element::dimension_tag U = DimensionTag,
            typename std::enable_if<
                U == specfem::element::dimension_tag::dim2>::type * = nullptr>
  KOKKOS_INLINE_FUNCTION void
  store_on_device(const int timestep, const IndexType index,
                  const PointSourceType &point_source) const {
    const auto isource = index.imap;
    for (int component = 0; component < components; component++) {
      source_time_function(timestep, isource, component) =
          point_source.stf(component);
      source_array(isource, component, index.iz, index.ix) =
          point_source.lagrange_interpolant(component);
    }
  }

  // ── Data access: store_on_device (dim3) ──────────────────────────────────

  template <typename IndexType, typename PointSourceType,
            specfem::element::dimension_tag U = DimensionTag,
            typename std::enable_if<
                U == specfem::element::dimension_tag::dim3>::type * = nullptr>
  KOKKOS_INLINE_FUNCTION void
  store_on_device(const int timestep, const IndexType index,
                  const PointSourceType &point_source) const {
    const auto isource = index.imap;
    for (int component = 0; component < components; component++) {
      source_time_function(timestep, isource, component) =
          point_source.stf(component);
      source_array(isource, component, index.iz, index.iy, index.ix) =
          point_source.lagrange_interpolant(component);
    }
  }

  // ── Data access: load_on_host (dim2) ─────────────────────────────────────

  template <typename IndexType, typename PointSourceType,
            specfem::element::dimension_tag U = DimensionTag,
            typename std::enable_if<
                U == specfem::element::dimension_tag::dim2>::type * = nullptr>
  void load_on_host(const int timestep, const IndexType index,
                    PointSourceType &point_source) const {
    const auto isource = index.imap;
    for (int component = 0; component < components; component++) {
      point_source.stf(component) =
          h_source_time_function(timestep, isource, component);
      point_source.lagrange_interpolant(component) =
          h_source_array(isource, component, index.iz, index.ix);
    }
  }

  // ── Data access: load_on_host (dim3) ─────────────────────────────────────

  template <typename IndexType, typename PointSourceType,
            specfem::element::dimension_tag U = DimensionTag,
            typename std::enable_if<
                U == specfem::element::dimension_tag::dim3>::type * = nullptr>
  void load_on_host(const int timestep, const IndexType index,
                    PointSourceType &point_source) const {
    const auto isource = index.imap;
    for (int component = 0; component < components; component++) {
      point_source.stf(component) =
          h_source_time_function(timestep, isource, component);
      point_source.lagrange_interpolant(component) =
          h_source_array(isource, component, index.iz, index.iy, index.ix);
    }
  }

  // ── Data access: store_on_host (dim2) ────────────────────────────────────

  template <typename IndexType, typename PointSourceType,
            specfem::element::dimension_tag U = DimensionTag,
            typename std::enable_if<
                U == specfem::element::dimension_tag::dim2>::type * = nullptr>
  void store_on_host(const int timestep, const IndexType index,
                     const PointSourceType &point_source) const {
    const auto isource = index.imap;
    for (int component = 0; component < components; component++) {
      h_source_time_function(timestep, isource, component) =
          point_source.stf(component);
      h_source_array(isource, component, index.iz, index.ix) =
          point_source.lagrange_interpolant(component);
    }
  }

  // ── Data access: store_on_host (dim3) ────────────────────────────────────

  template <typename IndexType, typename PointSourceType,
            specfem::element::dimension_tag U = DimensionTag,
            typename std::enable_if<
                U == specfem::element::dimension_tag::dim3>::type * = nullptr>
  void store_on_host(const int timestep, const IndexType index,
                     const PointSourceType &point_source) const {
    const auto isource = index.imap;
    for (int component = 0; component < components; component++) {
      h_source_time_function(timestep, isource, component) =
          point_source.stf(component);
      h_source_array(isource, component, index.iz, index.iy, index.ix) =
          point_source.lagrange_interpolant(component);
    }
  }
};

/**
 * @brief Filter and sort sources by medium type
 */
template <specfem::element::dimension_tag DimensionTag,
          specfem::element::medium_tag MediumTag>
std::tuple<
    std::vector<std::shared_ptr<specfem::sources::source<DimensionTag> > >,
    std::vector<int> >
sort_sources_per_medium(
    const std::vector<std::shared_ptr<specfem::sources::source<DimensionTag> > >
        &sources,
    const specfem::assembly::element_types<DimensionTag> &element_types,
    const specfem::assembly::mesh<DimensionTag> &mesh) {

  std::vector<std::shared_ptr<specfem::sources::source<DimensionTag> > >
      sorted_sources;
  std::vector<int> source_indices;

  for (int isource = 0; isource < (int)sources.size(); isource++) {
    const auto &source = sources[isource];
    if (source->get_medium_tag() == MediumTag) {
      sorted_sources.push_back(source);
      source_indices.push_back(isource);
    }
  }
  return std::make_tuple(sorted_sources, source_indices);
}

} // namespace specfem::assembly::sources_impl
