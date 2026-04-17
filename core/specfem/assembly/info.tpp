#include "info.hpp"
#include "specfem/assembly/info/impl/bounds.hpp"
#include "specfem/assembly/info/impl/compute.hpp"
#include "specfem/assembly/info/impl/scatter_minmax.hpp"
#include "specfem/assembly/info/impl/distances.hpp"
#include "specfem/enums.hpp"
#include "specfem/point.hpp"
#include "specfem/setup.hpp"
#include "specfem/assembly/info/impl/bounding_box.hpp"
#include "specfem/execution.hpp"
#include "specfem/parallel_configuration.hpp"
#include <Kokkos_Core.hpp>

#include <limits>

namespace specfem::assembly::info::impl {

/// @brief Container for all scatter min/max reducers used in Info computation
/// @tparam ndim Number of spatial dimensions for bounding box
template <specfem::element::dimension_tag DimensionTag>
struct InfoScatters {

  constexpr static int ndim =
      specfem::element::dimension<DimensionTag>::dim;

  using ScatterT = ScatterMinMax<type_real>;
  using ScatterBoundsT = ScatterMinMax<type_real, ndim>;

  // Global property bounds
  ScatterT vp{"vp"};
  ScatterT vs{"vs"};
  ScatterT v{"v"};
  ScatterT rho{"rho"};

  // Distance bounds
  ScatterT distance{"distance"};
  ScatterT gll_distance{"gll_distance"};

  // Derived quantities
  ScatterT minimum_period{"minimum_period"};
  ScatterT dt{"dt"};

  // Domain bounding box
  ScatterBoundsT domain_bounds{"specfem::assembly::info::domain_bounds_scatter",
                               ndim};

  // Per-element scatter views
  ScatterT element_v;
  ScatterT element_distance;
  ScatterT element_gll_distance;

  explicit InfoScatters(int nspec)
      : element_v("specfem::assembly::info::element_v", nspec),
        element_distance("specfem::assembly::info::element_distance", nspec),
        element_gll_distance("specfem::assembly::info::element_gll_distance",
                             nspec) {}
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
    InfoScatters<DimensionTag> &scatters) {

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
        specfem::point::properties<specfem::tags::Tags<dimension_tag, medium_tag, property_tag, false>>
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
        type_real min_period =
            compute_minimum_period(avg_distance, vmin);

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
    const specfem::assembly::element_types<dimension_tag> &element_types) {

  // Create all scatter reducers in a single struct
  info::impl::InfoScatters<dimension_tag> scatters(mesh.nspec);

  // Process each medium/property combination
  if constexpr (specfem::element::dimension_tag::dim2 == dimension_tag) {
    FOR_EACH_IN_PRODUCT(
        (DIMENSION_TAG(DIM2),
         MEDIUM_TAG(ELASTIC_PSV, ELASTIC_SH, ACOUSTIC, POROELASTIC,
                    ELASTIC_PSV_T),
         PROPERTY_TAG(ISOTROPIC, ANISOTROPIC, ISOTROPIC_COSSERAT)),
        {
          info::impl::process_medium_elements<_dimension_tag_, _medium_tag_,
                                              _property_tag_>(
              mesh, properties, element_types, scatters);
        };);
  } else {
    FOR_EACH_IN_PRODUCT(
        (DIMENSION_TAG(DIM3),
         MEDIUM_TAG(ELASTIC, ACOUSTIC),
         PROPERTY_TAG(ISOTROPIC)),
        {
          info::impl::process_medium_elements<_dimension_tag_, _medium_tag_,
                                              _property_tag_>(
              mesh, properties, element_types, scatters);
        };);
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
  this->domain_bounds =
      info::impl::BoundingBox<dimension_tag>(scatters.domain_bounds.get_all_bounds());

  auto period_bounds = scatters.minimum_period.get_bounds();
  this->largest_minimum_period = period_bounds.max;

  auto dt_bounds = scatters.dt.get_bounds();
  this->suggested_time_step = dt_bounds.min;
}
