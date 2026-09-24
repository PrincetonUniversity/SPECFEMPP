#include "info.hpp"
#include "specfem/assembly/info/impl/bounding_box.hpp"
#include "specfem/assembly/info/impl/bounds.hpp"
#include "specfem/assembly/info/impl/compute.hpp"
#include "specfem/assembly/info/impl/distances.hpp"
#include "specfem/assembly/info/impl/region_counts.hpp"
#include "specfem/assembly/info/impl/scatter_minmax.hpp"
#include "specfem/enums.hpp"
#include "specfem/execution.hpp"
#include "specfem/mpi.hpp"
#include "specfem/parallel_configuration.hpp"
#include "specfem/point.hpp"
#include "specfem/setup.hpp"
#include "specfem/tag_dispatch.hpp"
#include <Kokkos_Core.hpp>

#include <limits>

namespace specfem::assembly::info::impl {

/// @brief Container for all scatter min/max reducers used in Info computation
/// @tparam ndim Number of spatial dimensions for bounding box
template <specfem::element::dimension_tag DimensionTag> struct InfoScatters {

  constexpr static int ndim = specfem::element::dimension<DimensionTag>::dim;

  using ScatterT = ScatterMinMax<type_real>;
  using ScatterBoundsT = ScatterMinMax<type_real, ndim>;

  // Global property bounds
  ScatterT vp{ "vp" };
  ScatterT vs{ "vs" };
  ScatterT v{ "v" };
  ScatterT rho{ "rho" };

  // Distance bounds
  ScatterT distance{ "distance" };
  ScatterT gll_distance{ "gll_distance" };

  // Derived quantities
  ScatterT minimum_period{ "minimum_period" };
  ScatterT dt{ "dt" };

  // Domain bounding box
  ScatterBoundsT domain_bounds{
    "specfem::assembly::info::domain_bounds_scatter", ndim
  };

  // Per-element scatter views
  ScatterT element_v;
  ScatterT element_distance;
  ScatterT element_gll_distance;

  // Accumulate whole-mesh sums in double precision to limit roundoff error.
  Kokkos::View<double *> integrals;

  explicit InfoScatters(int nspec)
      : element_v("specfem::assembly::info::element_v", nspec),
        element_distance("specfem::assembly::info::element_distance", nspec),
        element_gll_distance("specfem::assembly::info::element_gll_distance",
                             nspec),
        integrals("specfem::assembly::info::integrals", 2) {
    Kokkos::deep_copy(integrals, 0.0);
  }
};

/// @brief Process elements of a specific medium/property type to gather info
/// @tparam DimensionTag The dimension type (dim2 or dim3)
/// @tparam MediumTag The medium type (elastic, acoustic, etc.)
/// @tparam PropertyTag The property type (isotropic, anisotropic, etc.)
template <specfem::element::dimension_tag DimensionTag,
          specfem::element::medium_tag MediumTag,
          specfem::element::property_tag PropertyTag>
void process_medium_elements(
    const specfem::assembly::mesh<DimensionTag> &mesh,
    const specfem::assembly::properties<DimensionTag> &properties,
    const specfem::assembly::element_types<DimensionTag> &element_types,
    const specfem::assembly::jacobian_matrix<DimensionTag> &jacobian_matrix,
    const bool compute_integrals, InfoScatters<DimensionTag> &scatters) {

  constexpr specfem::element::dimension_tag dimension_tag = DimensionTag;
  constexpr specfem::element::medium_tag medium_tag = MediumTag;
  constexpr specfem::element::property_tag property_tag = PropertyTag;
  auto elements =
      element_types.get_elements_on_device(medium_tag, property_tag);

  constexpr bool using_simd = false;
  using simd = specfem::datatype::simd<type_real, using_simd>;
  using parallel_config = specfem::parallel_configuration::default_chunk_config<
      dimension_tag, simd, Kokkos::DefaultExecutionSpace>;

  specfem::execution::ChunkedDomainIterator chunk(parallel_config(), elements,
                                                  mesh.element_grid);

  specfem::execution::for_all(
      "specfem::assembly::Info()", chunk,
      KOKKOS_LAMBDA(
          const typename decltype(chunk)::base_index_type &iterator_index) {
        const auto point_index = iterator_index.get_index();

        // Create point property object
        specfem::point::properties<
            specfem::tags::Tags<dimension_tag, medium_tag, property_tag, false>>
            point_property;
        specfem::assembly::load_on_device(point_index, properties,
                                          point_property);

        // Get scatter accessors
        auto vp_acc = scatters.vp.access();
        auto vs_acc = scatters.vs.access();
        auto v_acc = scatters.v.access();
        auto rho_acc = scatters.rho.access();
        auto distance_acc = scatters.distance.access();
        auto gll_distance_acc = scatters.gll_distance.access();

        // Per-element scatter accessors
        auto element_v_acc = scatters.element_v.access();
        auto element_distance_acc = scatters.element_distance.access();
        auto element_gll_distance_acc = scatters.element_gll_distance.access();

        // Compute the necessary properties
        auto rho_val = point_property.rho();
        auto vp_val = point_property.vp();
        auto vs_val = point_property.vs();
        auto vmax = point_property.vmax();
        auto vmin = point_property.vmin();

        if (compute_integrals) {
          specfem::point::jacobian_matrix<dimension_tag, true, false>
              point_jacobian;
          specfem::assembly::load_on_device(point_index, jacobian_matrix,
                                            point_jacobian);
          specfem::point::weights<dimension_tag> point_weights;
          specfem::assembly::load_on_device(point_index, mesh.weights,
                                            point_weights);
          const double weighted_jacobian =
              static_cast<double>(point_jacobian.jacobian()) *
              static_cast<double>(point_weights.product());
          Kokkos::atomic_add(&scatters.integrals(0), weighted_jacobian);
          Kokkos::atomic_add(&scatters.integrals(1),
                             static_cast<double>(rho_val) * weighted_jacobian);
        }

        // Update global min/max
        vp_acc.update(vp_val);
        vs_acc.update(vs_val);
        v_acc.update_min(vmin);
        v_acc.update_max(vmax);
        rho_acc.update(rho_val);
        element_v_acc.update_min(point_index.ispec, vmin);
        element_v_acc.update_max(point_index.ispec, vmax);

        specfem::point::global_coordinates<dimension_tag> current_point;
        specfem::assembly::load_on_device(point_index, mesh, current_point);

        // Update bounding box
        auto domain_bounds_acc = scatters.domain_bounds.access();
        domain_bounds_acc.update(current_point.coordinates());

        // Compute distances between adjacent GLL points
        compute_gll_distances<dimension_tag>(point_index, mesh, current_point,
                                             gll_distance_acc,
                                             element_gll_distance_acc);

        // Compute element sizes (corner-to-corner distances)
        compute_element_sizes<dimension_tag>(point_index, mesh, current_point,
                                             distance_acc,
                                             element_distance_acc);
      });

  // Finalize reductions
  scatters.vp.contribute();
  scatters.vs.contribute();
  scatters.v.contribute();
  scatters.rho.contribute();
  scatters.distance.contribute();
  scatters.gll_distance.contribute();
  scatters.element_v.contribute();
  scatters.element_distance.contribute();
  scatters.element_gll_distance.contribute();
  Kokkos::fence();
}

/// @brief Finalize element min/max by computing derived quantities
/// @tparam DimensionTag The dimension type (dim2 or dim3)
template <specfem::element::dimension_tag DimensionTag>
void finalize_element_minmax(const specfem::assembly::mesh<DimensionTag> &mesh,
                             InfoScatters<DimensionTag> &scatters) {
  Kokkos::parallel_for(
      "specfem::assembly::Info::finalize_element_minmax",
      Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, mesh.nspec),
      KOKKOS_LAMBDA(const int &ispec) {
        // Get min and max velocities for the element
        auto vmin = scatters.element_v.min_view[ispec];
        auto vmax = scatters.element_v.max_view[ispec];

        // Get max distance between GLL points for the element
        auto distance_max = scatters.element_distance.max_view[ispec];
        auto gll_distance_min = scatters.element_gll_distance.min_view[ispec];

        // Get scatter accessors for the reductions
        auto minimum_period_acc = scatters.minimum_period.access();
        auto dt_acc = scatters.dt.access();

        // Compute average GLL distance from element size
        int fgll = mesh.element_grid.ngll - 1;
        type_real avg_distance =
            compute_average_gll_spacing(distance_max, fgll);
        type_real min_period = compute_minimum_period(avg_distance, vmin);

        // Update minimum period scatter
        minimum_period_acc.update(min_period);

        // Suggested time step based on CFL condition
        type_real element_dt =
            compute_suggested_timestep(gll_distance_min, vmax);
        dt_acc.update(element_dt);
      });
  scatters.minimum_period.contribute();
  scatters.dt.contribute();
  Kokkos::fence();
}

} // namespace specfem::assembly::info::impl

template <specfem::element::dimension_tag DimensionTag>
specfem::assembly::Info<DimensionTag>::Info(
    const specfem::assembly::mesh<dimension_tag> &mesh,
    const specfem::assembly::properties<dimension_tag> &properties,
    const specfem::assembly::element_types<dimension_tag> &element_types,
    const specfem::assembly::jacobian_matrix<dimension_tag> &jacobian_matrix) {

  // Create all scatter reducers in a single struct
  info::impl::InfoScatters<dimension_tag> scatters(mesh.nspec);

  const bool has_element_context = element_types.has_element_context();

  // Process each medium/property combination
  if constexpr (specfem::element::dimension_tag::dim2 == dimension_tag) {
    specfem::tag_dispatch::for_each(
        DIMENSION_SET(dim2) *
            MEDIUM_SET(elastic_psv, elastic_sh, acoustic, poroelastic,
                       elastic_psv_t) *
            PROPERTY_SET(isotropic, anisotropic, isotropic_cosserat) *
            ATTENUATION_SET(none),
        [&]<typename ElementTags>() {
          info::impl::process_medium_elements<ElementTags::dimension_tag,
                                              ElementTags::medium_tag,
                                              ElementTags::property_tag>(
              mesh, properties, element_types, jacobian_matrix,
              has_element_context, scatters);
        });
  } else {
    specfem::tag_dispatch::for_each(
        DIMENSION_SET(dim3) * MEDIUM_SET(elastic, acoustic, elastic_spin) *
            PROPERTY_SET(isotropic, anisotropic, isotropic_cosserat) *
            ATTENUATION_SET(none),
        [&]<typename ElementTags>() {
          info::impl::process_medium_elements<ElementTags::dimension_tag,
                                              ElementTags::medium_tag,
                                              ElementTags::property_tag>(
              mesh, properties, element_types, jacobian_matrix,
              has_element_context, scatters);
        });
  };

  // Compute derived quantities (minimum period, dt) from per-element data
  info::impl::finalize_element_minmax(mesh, scatters);

  // Copy results to member variables
  this->vp = scatters.vp.get_bounds();
  this->vs = scatters.vs.get_bounds();
  this->v = scatters.v.get_bounds();
  this->rho = scatters.rho.get_bounds();
  this->element_size = scatters.distance.get_bounds();
  this->gll_distance = scatters.gll_distance.get_bounds();
  this->domain_bounds = info::impl::BoundingBox<dimension_tag>(
      scatters.domain_bounds.get_all_bounds());

  auto period_bounds = scatters.minimum_period.get_bounds();
  this->largest_minimum_period = period_bounds.max;

  auto dt_bounds = scatters.dt.get_bounds();
  this->suggested_time_step = dt_bounds.min;

  int local_gll_element = -1;
  int local_dt_element = -1;
  type_real local_gll_min = std::numeric_limits<type_real>::max();
  type_real local_dt_min = std::numeric_limits<type_real>::max();
  if (has_element_context) {
    const auto element_v = scatters.element_v.get_all_bounds();
    const auto element_gll = scatters.element_gll_distance.get_all_bounds();
    const type_real finite_max = std::numeric_limits<type_real>::max();
    const type_real finite_min = std::numeric_limits<type_real>::lowest();

    for (const auto region : { specfem::element::region_tag::crust_mantle,
                               specfem::element::region_tag::outer_core,
                               specfem::element::region_tag::inner_core }) {
      typename Info<dimension_tag>::RegionInfo region_info;
      region_info.radius = { finite_max, finite_min };
      region_info.gll_distance = { finite_max, finite_min };
      region_info.v = { finite_max, finite_min };
      region_info.suggested_time_step = finite_max;
      this->regions.emplace(region, region_info);
    }

    for (int ispec = 0; ispec < mesh.nspec; ++ispec) {
      auto &region = this->regions.at(element_types.regions(ispec));
      ++region.element_count;
      region.radius.min =
          Kokkos::min(region.radius.min, element_types.rmin(ispec));
      region.radius.max =
          Kokkos::max(region.radius.max, element_types.rmax(ispec));
      region.gll_distance.min =
          Kokkos::min(region.gll_distance.min, element_gll[ispec].min);
      region.gll_distance.max =
          Kokkos::max(region.gll_distance.max, element_gll[ispec].max);
      region.v.min = Kokkos::min(region.v.min, element_v[ispec].min);
      region.v.max = Kokkos::max(region.v.max, element_v[ispec].max);
      const type_real element_dt = info::impl::compute_suggested_timestep(
          element_gll[ispec].min, element_v[ispec].max);
      region.suggested_time_step =
          Kokkos::min(region.suggested_time_step, element_dt);

      if (element_gll[ispec].min < local_gll_min) {
        local_gll_min = element_gll[ispec].min;
        local_gll_element = ispec;
      }
      if (element_dt < local_dt_min) {
        local_dt_min = element_dt;
        local_dt_element = ispec;
      }
    }

    const auto integrals = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), scatters.integrals);
    this->total_volume = integrals(0);
    this->total_mass = integrals(1);
  }

  // Reduce bounds across MPI ranks
  const auto comm = specfem::MPI::communicator();
  SPECFEM_MPI_SAFECALL(MPI_Allreduce(MPI_IN_PLACE, &this->vp.min, 1,
                                     SPECFEM_MPI_TYPE_REAL, MPI_MIN, comm));
  SPECFEM_MPI_SAFECALL(MPI_Allreduce(MPI_IN_PLACE, &this->vp.max, 1,
                                     SPECFEM_MPI_TYPE_REAL, MPI_MAX, comm));
  SPECFEM_MPI_SAFECALL(MPI_Allreduce(MPI_IN_PLACE, &this->vs.min, 1,
                                     SPECFEM_MPI_TYPE_REAL, MPI_MIN, comm));
  SPECFEM_MPI_SAFECALL(MPI_Allreduce(MPI_IN_PLACE, &this->vs.max, 1,
                                     SPECFEM_MPI_TYPE_REAL, MPI_MAX, comm));
  SPECFEM_MPI_SAFECALL(MPI_Allreduce(MPI_IN_PLACE, &this->v.min, 1,
                                     SPECFEM_MPI_TYPE_REAL, MPI_MIN, comm));
  SPECFEM_MPI_SAFECALL(MPI_Allreduce(MPI_IN_PLACE, &this->v.max, 1,
                                     SPECFEM_MPI_TYPE_REAL, MPI_MAX, comm));
  SPECFEM_MPI_SAFECALL(MPI_Allreduce(MPI_IN_PLACE, &this->rho.min, 1,
                                     SPECFEM_MPI_TYPE_REAL, MPI_MIN, comm));
  SPECFEM_MPI_SAFECALL(MPI_Allreduce(MPI_IN_PLACE, &this->rho.max, 1,
                                     SPECFEM_MPI_TYPE_REAL, MPI_MAX, comm));
  SPECFEM_MPI_SAFECALL(MPI_Allreduce(MPI_IN_PLACE, &this->element_size.min, 1,
                                     SPECFEM_MPI_TYPE_REAL, MPI_MIN, comm));
  SPECFEM_MPI_SAFECALL(MPI_Allreduce(MPI_IN_PLACE, &this->element_size.max, 1,
                                     SPECFEM_MPI_TYPE_REAL, MPI_MAX, comm));
  SPECFEM_MPI_SAFECALL(MPI_Allreduce(MPI_IN_PLACE, &this->gll_distance.min, 1,
                                     SPECFEM_MPI_TYPE_REAL, MPI_MIN, comm));
  SPECFEM_MPI_SAFECALL(MPI_Allreduce(MPI_IN_PLACE, &this->gll_distance.max, 1,
                                     SPECFEM_MPI_TYPE_REAL, MPI_MAX, comm));
  constexpr int ndim = info::impl::InfoScatters<dimension_tag>::ndim;
  for (int i = 0; i < ndim; ++i) {
    SPECFEM_MPI_SAFECALL(
        MPI_Allreduce(MPI_IN_PLACE, &this->domain_bounds.bounds_array(i).min, 1,
                      SPECFEM_MPI_TYPE_REAL, MPI_MIN, comm));
    SPECFEM_MPI_SAFECALL(
        MPI_Allreduce(MPI_IN_PLACE, &this->domain_bounds.bounds_array(i).max, 1,
                      SPECFEM_MPI_TYPE_REAL, MPI_MAX, comm));
  }
  SPECFEM_MPI_SAFECALL(MPI_Allreduce(MPI_IN_PLACE,
                                     &this->largest_minimum_period, 1,
                                     SPECFEM_MPI_TYPE_REAL, MPI_MAX, comm));
  SPECFEM_MPI_SAFECALL(MPI_Allreduce(MPI_IN_PLACE, &this->suggested_time_step,
                                     1, SPECFEM_MPI_TYPE_REAL, MPI_MIN, comm));

  if (has_element_context) {
    for (auto &[region_tag, region] : this->regions) {
      SPECFEM_MPI_SAFECALL(MPI_Allreduce(MPI_IN_PLACE, &region.element_count, 1,
                                         MPI_INT, MPI_SUM, comm));
      SPECFEM_MPI_SAFECALL(MPI_Allreduce(MPI_IN_PLACE, &region.radius.min, 1,
                                         SPECFEM_MPI_TYPE_REAL, MPI_MIN, comm));
      SPECFEM_MPI_SAFECALL(MPI_Allreduce(MPI_IN_PLACE, &region.radius.max, 1,
                                         SPECFEM_MPI_TYPE_REAL, MPI_MAX, comm));
      SPECFEM_MPI_SAFECALL(MPI_Allreduce(MPI_IN_PLACE, &region.gll_distance.min,
                                         1, SPECFEM_MPI_TYPE_REAL, MPI_MIN,
                                         comm));
      SPECFEM_MPI_SAFECALL(MPI_Allreduce(MPI_IN_PLACE, &region.gll_distance.max,
                                         1, SPECFEM_MPI_TYPE_REAL, MPI_MAX,
                                         comm));
      SPECFEM_MPI_SAFECALL(MPI_Allreduce(MPI_IN_PLACE, &region.v.min, 1,
                                         SPECFEM_MPI_TYPE_REAL, MPI_MIN, comm));
      SPECFEM_MPI_SAFECALL(MPI_Allreduce(MPI_IN_PLACE, &region.v.max, 1,
                                         SPECFEM_MPI_TYPE_REAL, MPI_MAX, comm));
      SPECFEM_MPI_SAFECALL(MPI_Allreduce(MPI_IN_PLACE,
                                         &region.suggested_time_step, 1,
                                         SPECFEM_MPI_TYPE_REAL, MPI_MIN, comm));
      this->elements_per_region.emplace(region_tag, region.element_count);
    }

    SPECFEM_MPI_SAFECALL(MPI_Allreduce(MPI_IN_PLACE, &this->total_volume, 1,
                                       MPI_DOUBLE, MPI_SUM, comm));
    SPECFEM_MPI_SAFECALL(MPI_Allreduce(MPI_IN_PLACE, &this->total_mass, 1,
                                       MPI_DOUBLE, MPI_SUM, comm));

    const int rank = specfem::MPI::get_rank();
    const auto locate_extremum = [&](const int local_element,
                                     const bool owns_global_extremum,
                                     ExtremumLocation &location) {
      int owner = owns_global_extremum ? rank : std::numeric_limits<int>::max();
      SPECFEM_MPI_SAFECALL(
          MPI_Allreduce(MPI_IN_PLACE, &owner, 1, MPI_INT, MPI_MIN, comm));
      location.rank = owner;
      if (rank == owner) {
        location.element = local_element;
        location.radius = { element_types.rmin(local_element),
                            element_types.rmax(local_element) };
      }
      SPECFEM_MPI_SAFECALL(
          MPI_Bcast(&location.element, 1, MPI_INT, owner, comm));
      SPECFEM_MPI_SAFECALL(MPI_Bcast(&location.radius.min, 1,
                                     SPECFEM_MPI_TYPE_REAL, owner, comm));
      SPECFEM_MPI_SAFECALL(MPI_Bcast(&location.radius.max, 1,
                                     SPECFEM_MPI_TYPE_REAL, owner, comm));
    };
    locate_extremum(local_gll_element, local_gll_min == this->gll_distance.min,
                    this->minimum_gll_distance_location);
    locate_extremum(local_dt_element, local_dt_min == this->suggested_time_step,
                    this->cfl_limit_location);
  }
}
