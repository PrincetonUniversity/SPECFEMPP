#pragma once

#include "specfem/assembly/info.hpp"
#include "specfem/assembly/info/impl/compute.hpp"
#include "specfem/assembly/info/impl/distances.hpp"
#include "specfem/mpi.hpp"
#include "specfem/point.hpp"
#include "specfem/tag_dispatch.hpp"
#include <algorithm>
#include <limits>

namespace specfem::assembly::info_impl {

using GlobeElementIntegrals = Kokkos::View<double *[5]>;

/**
 * @brief Single-element accumulator for @c info::impl::compute_gll_distances.
 *
 * Satisfies both the global and per-element accumulator interfaces, keeping
 * only the shortest adjacent GLL edge seen so far.
 */
struct MinimumEdge {
  type_real distance =
      Kokkos::Experimental::finite_max_v<type_real>; ///< Shortest edge, m.

  /** @brief Record an edge length. @param value Edge length. */
  KOKKOS_INLINE_FUNCTION void update(const type_real value) {
    distance = Kokkos::min(distance, value);
  }

  /** @brief Record an edge length; the element index is implied.
   *  @param value Edge length. */
  KOKKOS_INLINE_FUNCTION void update(const int, const type_real value) {
    update(value);
  }
};

/**
 * @brief Integrate each element in double precision, independently on device.
 *
 * Columns contain volume, mass, minimum/maximum GLL radius and the radius of
 * the first endpoint of the shortest GLL edge. Shared GLL nodes contribute to
 * each incident element's quadrature, as required for a volume integral.
 * @tparam ElementTags Dimension, medium and property tags.
 * @param mesh Assembled physical coordinates and quadrature weights.
 * @param properties Material properties in SI units.
 * @param element_types Compute-domain medium and property classification.
 * @param jacobians Existing physical Jacobian determinants.
 * @param integrals Output per-element integrals and radial diagnostics.
 */
template <typename ElementTags>
void integrate_globe_elements(
    const specfem::assembly::mesh<ElementTags::dimension_tag> &mesh,
    const specfem::assembly::properties<ElementTags::dimension_tag> &properties,
    const specfem::assembly::element_types<ElementTags::dimension_tag>
        &element_types,
    const specfem::assembly::jacobian_matrix<ElementTags::dimension_tag>
        &jacobians,
    const GlobeElementIntegrals &integrals) {
  constexpr auto dimension = ElementTags::dimension_tag;
  const auto elements = element_types.get_elements_on_device(
      ElementTags::medium_tag, ElementTags::property_tag);
  const auto weights = mesh.weights;
  const auto jacobian = jacobians.jacobian;
  const auto grid = mesh.element_grid;
  Kokkos::parallel_for(
      "specfem::assembly::Info::globe_integrals",
      Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(elements.begin_index(),
                                                         elements.end_index()),
      KOKKOS_LAMBDA(const int ispec) {
        double volume = 0, mass = 0, radius_at_minimum = 0;
        double minimum_radius = Kokkos::Experimental::finite_max_v<double>;
        double maximum_radius = 0;
        MinimumEdge edge;
        for (int iz = 0; iz < grid.ngllz; ++iz) {
          for (int iy = 0; iy < grid.nglly; ++iy) {
            for (int ix = 0; ix < grid.ngllx; ++ix) {
              const specfem::point::index<dimension, false> index(ispec, iz, iy,
                                                                  ix);
              specfem::point::properties<
                  specfem::tags::Tags<dimension, ElementTags::medium_tag,
                                      ElementTags::property_tag, false>>
                  point_property;
              specfem::assembly::load_on_device(index, properties,
                                                point_property);
              const double dv =
                  static_cast<double>(jacobian(ispec, iz, iy, ix)) *
                  weights(iz) * weights(iy) * weights(ix);
              volume += dv;
              mass += static_cast<double>(point_property.rho()) * dv;
              specfem::point::global_coordinates<dimension> point;
              specfem::assembly::load_on_device(index, mesh, point);
              const double radius =
                  Kokkos::sqrt(static_cast<double>(point.x) * point.x +
                               static_cast<double>(point.y) * point.y +
                               static_cast<double>(point.z) * point.z);
              minimum_radius = Kokkos::fmin(minimum_radius, radius);
              maximum_radius = Kokkos::fmax(maximum_radius, radius);
              // Reuse the Info edge kernel so this radius belongs to the edge
              // whose length the element_gll_distance scatter reports.
              const type_real previous = edge.distance;
              info::impl::compute_gll_distances<dimension>(index, mesh, point,
                                                           edge, edge);
              if (edge.distance < previous) {
                radius_at_minimum = radius;
              }
            }
          }
        }
        integrals(ispec, 0) = volume;
        integrals(ispec, 1) = mass;
        integrals(ispec, 2) = minimum_radius;
        integrals(ispec, 3) = maximum_radius;
        integrals(ispec, 4) = radius_at_minimum;
      });
}

/**
 * @brief Reduce an extremum's location; break ties by rank then element.
 *
 * The owner is found by comparing @p local_value with the minimum of the same
 * values across ranks, so exact equality holds on the owning rank regardless
 * of where (host or device) the values were computed. If no rank owns an
 * element, @p location is reset on every rank.
 * @param local_value This rank's minimum; ranks without elements pass the
 *        largest representable value.
 * @param location Local winner on input, global winner on output.
 */
inline void reduce_location(type_real local_value, InfoLocation &location) {
  const auto comm = specfem::MPI::communicator();
  type_real global_value = local_value;
  SPECFEM_MPI_SAFECALL(MPI_Allreduce(MPI_IN_PLACE, &global_value, 1,
                                     SPECFEM_MPI_TYPE_REAL, MPI_MIN, comm));
  int owner = local_value == global_value && location.compute_element >= 0
                  ? specfem::MPI::get_rank()
                  : std::numeric_limits<int>::max();
  SPECFEM_MPI_SAFECALL(
      MPI_Allreduce(MPI_IN_PLACE, &owner, 1, MPI_INT, MPI_MIN, comm));
  if (owner == std::numeric_limits<int>::max()) {
    location = InfoLocation{};
    return;
  }
  location.rank = owner;
  int indices[] = { location.compute_element, location.mesh_element,
                    static_cast<int>(location.region) };
  SPECFEM_MPI_SAFECALL(MPI_Bcast(indices, 3, MPI_INT, owner, comm));
  SPECFEM_MPI_SAFECALL(MPI_Bcast(&location.radius, 1, MPI_DOUBLE, owner, comm));
  location.compute_element = indices[0];
  location.mesh_element = indices[1];
  location.region = static_cast<specfem::element::region_tag>(indices[2]);
}

/**
 * @brief Summarize globe regions using the existing per-element CFL reductions.
 * @tparam Scatters Existing Info scatter reducers.
 * @param mesh Assembled mesh for this rank.
 * @param properties Material properties in SI units.
 * @param element_types Globe region context in compute order.
 * @param jacobians Existing physical Jacobian determinants.
 * @param scatters Existing per-element spacing and wave speed reductions.
 * @param result Receives globally reduced globe statistics.
 */
template <typename Scatters>
void compute_globe_info(
    const specfem::assembly::mesh<specfem::element::dimension_tag::dim3> &mesh,
    const specfem::assembly::properties<specfem::element::dimension_tag::dim3>
        &properties,
    const specfem::assembly::element_types<
        specfem::element::dimension_tag::dim3> &element_types,
    const specfem::assembly::jacobian_matrix<
        specfem::element::dimension_tag::dim3> &jacobians,
    const Scatters &scatters,
    Info<specfem::element::dimension_tag::dim3> &result) {
  const GlobeElementIntegrals integrals("globe_element_integrals", mesh.nspec);
  specfem::tag_dispatch::for_each(
      DIMENSION_SET(dim3) * MEDIUM_SET(elastic, acoustic, elastic_spin) *
          PROPERTY_SET(isotropic, anisotropic, isotropic_cosserat) *
          ATTENUATION_SET(none),
      [&]<typename ElementTags>() {
        integrate_globe_elements<ElementTags>(mesh, properties, element_types,
                                              jacobians, integrals);
      });
  const auto host_integrals =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), integrals);
  const auto velocities = scatters.element_v.get_all_bounds();
  const auto distances = scatters.element_gll_distance.get_all_bounds();
  // Use the very same device results as the global CFL reduction, including
  // its rounding, so the limiting location describes the reported time step.
  const auto time_steps = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace(), scatters.element_dt);
  constexpr auto maximum = std::numeric_limits<type_real>::max();
  type_real local_spacing = maximum, local_dt = maximum;
  for (const auto region : { specfem::element::region_tag::crust_mantle,
                             specfem::element::region_tag::outer_core,
                             specfem::element::region_tag::inner_core }) {
    RegionInfo statistics;
    statistics.radius = { maximum, -maximum };
    statistics.wave_speed = { maximum, -maximum };
    statistics.minimum_gll_spacing = maximum;
    statistics.suggested_time_step = maximum;
    result.regions.emplace(region, statistics);
  }
  for (int ispec = 0; ispec < mesh.nspec; ++ispec) {
    const auto region = element_types.get_region_tag(ispec);
    auto &statistics = result.regions.at(region);
    ++statistics.element_count;
    statistics.radius.min =
        std::min(statistics.radius.min,
                 static_cast<type_real>(host_integrals(ispec, 2)));
    statistics.radius.max =
        std::max(statistics.radius.max,
                 static_cast<type_real>(host_integrals(ispec, 3)));
    statistics.wave_speed.min =
        std::min(statistics.wave_speed.min, velocities[ispec].min);
    statistics.wave_speed.max =
        std::max(statistics.wave_speed.max, velocities[ispec].max);
    const auto spacing = distances[ispec].min;
    const auto dt = time_steps(ispec);
    statistics.minimum_gll_spacing =
        std::min(statistics.minimum_gll_spacing, spacing);
    statistics.suggested_time_step =
        std::min(statistics.suggested_time_step, dt);
    const InfoLocation location{ specfem::MPI::get_rank(), ispec,
                                 mesh.h_compute_to_mesh(ispec), region,
                                 host_integrals(ispec, 4) };
    if (spacing < local_spacing) {
      local_spacing = spacing;
      result.minimum_spacing_location = location;
    }
    if (dt < local_dt) {
      local_dt = dt;
      result.cfl_location = location;
    }
    result.total_volume += host_integrals(ispec, 0);
    result.total_mass += host_integrals(ispec, 1);
  }
  const auto comm = specfem::MPI::communicator();
  for (auto &[region, statistics] : result.regions) {
    SPECFEM_MPI_SAFECALL(MPI_Allreduce(MPI_IN_PLACE, &statistics.element_count,
                                       1, MPI_INT, MPI_SUM, comm));
    for (auto *value :
         { &statistics.radius.min, &statistics.wave_speed.min,
           &statistics.minimum_gll_spacing, &statistics.suggested_time_step }) {
      SPECFEM_MPI_SAFECALL(MPI_Allreduce(MPI_IN_PLACE, value, 1,
                                         SPECFEM_MPI_TYPE_REAL, MPI_MIN, comm));
    }
    for (auto *value : { &statistics.radius.max, &statistics.wave_speed.max }) {
      SPECFEM_MPI_SAFECALL(MPI_Allreduce(MPI_IN_PLACE, value, 1,
                                         SPECFEM_MPI_TYPE_REAL, MPI_MAX, comm));
    }
    if (statistics.element_count == 0) {
      statistics = RegionInfo{};
    }
    result.elements_per_region[region] = statistics.element_count;
  }
  SPECFEM_MPI_SAFECALL(MPI_Allreduce(MPI_IN_PLACE, &result.total_volume, 1,
                                     MPI_DOUBLE, MPI_SUM, comm));
  SPECFEM_MPI_SAFECALL(MPI_Allreduce(MPI_IN_PLACE, &result.total_mass, 1,
                                     MPI_DOUBLE, MPI_SUM, comm));
  reduce_location(local_spacing, result.minimum_spacing_location);
  reduce_location(local_dt, result.cfl_location);
}

} // namespace specfem::assembly::info_impl
