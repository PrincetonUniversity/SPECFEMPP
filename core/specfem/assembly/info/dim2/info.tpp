#include "info.hpp"
#include "specfem/assembly/assembly.hpp"
#include "specfem/point.hpp"
#include "specfem_setup.hpp"
#include "specfem/assembly/info/impl/bounds.hpp"
#include "specfem/assembly/info/impl/compute.hpp"
#include "impl/bounding_box.hpp"
#include <limits>

specfem::assembly::Info<specfem::dimension::type::dim2>::
Info(specfem::assembly::assembly<dimension_tag> &assembly) {
  
  // Get reference to element types, and mesh
  const auto &element_types = assembly.element_types;
  const auto &mesh = assembly.mesh;

  // Domain bounds
  domain_bounds = info::impl::BoundingBox<dimension_tag>(
      mesh.xmin, mesh.xmax, mesh.zmin, mesh.zmax);
  

  // Views to store min and max results (single element each)
  Kokkos::View<type_real[1]> min_vp("min_vp");
  Kokkos::View<type_real[1]> max_vp("max_vp");

  Kokkos::View<type_real[1]> min_vs("min_vs");
  Kokkos::View<type_real[1]> max_vs("max_vs");

  Kokkos::View<type_real[1]> min_v("min_v");
  Kokkos::View<type_real[1]> max_v("max_v");

  Kokkos::View<type_real[1]> min_rho("min_rho");
  Kokkos::View<type_real[1]> max_rho("max_rho");

  Kokkos::View<type_real[1]> min_distance("min_distance");
  Kokkos::View<type_real[1]> max_distance("max_distance");

  Kokkos::View<type_real[1]> min_gll_distance("min_gll_distance");
  Kokkos::View<type_real[1]> max_gll_distance("max_gll_distance");

  Kokkos::View<type_real[1]> max_minimum_period("max_minimum_period");
  Kokkos::View<type_real[1]> dt_suggested("dt_suggested");

  // Initialize min_result to max double, max_result to min double
  Kokkos::deep_copy(min_vp, std::numeric_limits<type_real>::max());
  Kokkos::deep_copy(max_vp, std::numeric_limits<type_real>::lowest());

  Kokkos::deep_copy(min_vs, std::numeric_limits<type_real>::max());
  Kokkos::deep_copy(max_vs, std::numeric_limits<type_real>::lowest());

  Kokkos::deep_copy(min_v, std::numeric_limits<type_real>::max());
  Kokkos::deep_copy(max_v, std::numeric_limits<type_real>::lowest());

  Kokkos::deep_copy(min_rho, std::numeric_limits<type_real>::max());
  Kokkos::deep_copy(max_rho, std::numeric_limits<type_real>::lowest());

  Kokkos::deep_copy(min_distance, std::numeric_limits<type_real>::max());
  Kokkos::deep_copy(max_distance, std::numeric_limits<type_real>::lowest());

  Kokkos::deep_copy(min_gll_distance, std::numeric_limits<type_real>::max());
  Kokkos::deep_copy(max_gll_distance, std::numeric_limits<type_real>::lowest());

  Kokkos::deep_copy(max_minimum_period, std::numeric_limits<type_real>::lowest());

  Kokkos::deep_copy(dt_suggested, std::numeric_limits<type_real>::max());

  // Create ScatterViews with ScatterMin and ScatterMax operations
  auto scatter_vp_min = Kokkos::Experimental::create_scatter_view<
      Kokkos::Experimental::ScatterMin>(min_vp);
  auto scatter_vp_max = Kokkos::Experimental::create_scatter_view<
      Kokkos::Experimental::ScatterMax>(max_vp);

  auto scatter_vs_min = Kokkos::Experimental::create_scatter_view<
      Kokkos::Experimental::ScatterMin>(min_vs);
  auto scatter_vs_max = Kokkos::Experimental::create_scatter_view<
      Kokkos::Experimental::ScatterMax>(max_vs);

  auto scatter_min_v = Kokkos::Experimental::create_scatter_view<
      Kokkos::Experimental::ScatterMin>(min_v);
  auto scatter_max_v = Kokkos::Experimental::create_scatter_view<
      Kokkos::Experimental::ScatterMax>(max_v);

  auto scatter_rho_min = Kokkos::Experimental::create_scatter_view<
      Kokkos::Experimental::ScatterMin>(min_rho);
  auto scatter_rho_max = Kokkos::Experimental::create_scatter_view<
      Kokkos::Experimental::ScatterMax>(max_rho);

  auto scatter_distance_min = Kokkos::Experimental::create_scatter_view<
      Kokkos::Experimental::ScatterMin>(min_distance);
  auto scatter_distance_max = Kokkos::Experimental::create_scatter_view<
      Kokkos::Experimental::ScatterMax>(max_distance);

  auto scatter_gll_distance_min = Kokkos::Experimental::create_scatter_view<
      Kokkos::Experimental::ScatterMin>(min_gll_distance);
  auto scatter_gll_distance_max = Kokkos::Experimental::create_scatter_view<
      Kokkos::Experimental::ScatterMax>(max_gll_distance);

  auto scatter_minimum_period_max = Kokkos::Experimental::create_scatter_view<
      Kokkos::Experimental::ScatterMax>(max_minimum_period);
      
  auto scatter_dt_suggested = Kokkos::Experimental::create_scatter_view<
      Kokkos::Experimental::ScatterMin>(dt_suggested);

  // Type alias
  using global_coord_type =
      specfem::point::global_coordinates<specfem::dimension::type::dim2>;

  FOR_EACH_IN_PRODUCT(
    (DIMENSION_TAG(DIM2),
      MEDIUM_TAG(ELASTIC_PSV, ELASTIC_SH, ACOUSTIC, POROELASTIC,
                ELASTIC_PSV_T),
      PROPERTY_TAG(ISOTROPIC, ANISOTROPIC, ISOTROPIC_COSSERAT)),
    {
      const auto dimension_tag = _dimension_tag_;
      const auto medium_tag = _medium_tag_;
      const auto property_tag = _property_tag_;

      auto elements = element_types.get_elements_on_host(
          medium_tag, property_tag);
      
      // Compute min and max using scatter reduce
      Kokkos::parallel_for("compute_minmax", elements.extent(0), KOKKOS_LAMBDA(int i) {
          
        auto ispec = elements(i);

        auto access_min_vp = scatter_vp_min.access();
        auto access_max_vp = scatter_vp_max.access();

        auto access_min_vs = scatter_vs_min.access();
        auto access_max_vs = scatter_vs_max.access();

        auto access_min_v = scatter_min_v.access();
        auto access_max_v = scatter_max_v.access();

        auto access_min_rho = scatter_rho_min.access();
        auto access_max_rho = scatter_rho_max.access();

        auto access_min_distance = scatter_distance_min.access();
        auto access_max_distance = scatter_distance_max.access();

        auto access_min_gll_distance = scatter_gll_distance_min.access();
        auto access_max_gll_distance = scatter_gll_distance_max.access();

        auto access_max_minimum_period = scatter_minimum_period_max.access();
        auto access_dt_suggested = scatter_dt_suggested.access();

        // Create point property object
        specfem::point::properties<dimension_tag, medium_tag, property_tag, false> point_property;

        // Element-local tracking for per-element calculations
        type_real element_min_v = std::numeric_limits<type_real>::max();
        type_real element_max_v = std::numeric_limits<type_real>::lowest();
        type_real element_min_gll_distance = std::numeric_limits<type_real>::max();
        type_real element_max_distance = std::numeric_limits<type_real>::lowest();

        // Get the material properties for this element
        for (int iz = 0; iz < mesh.element_grid.ngll; iz++) {
          for (int ix = 0; ix < mesh.element_grid.ngll; ix++) {

            // Create point index
            specfem::point::index<dimension_tag> point_index(ispec, ix, iz);

            // Load point properties
            specfem::assembly::load_on_host(point_index, assembly.properties, point_property);

            // Compute the necessary properties
            auto rho = point_property.rho();
            auto vp = point_property.vp();
            auto vs = point_property.vs();
            auto vmax = point_property.vmax();
            auto vmin = point_property.vmin();

            // Update min and max for vp
            access_min_vp(0).update(vp);
            access_max_vp(0).update(vp);

            // Update min and max for vs
            access_min_vs(0).update(vs);
            access_max_vs(0).update(vs);

            // Update min and max for v (global scatter)
            access_min_v(0).update(vmin);
            access_max_v(0).update(vmax);

            // Update element-local min/max velocities
            element_min_v = Kokkos::fmin(element_min_v, vmin);
            element_max_v = Kokkos::fmax(element_max_v, vmax);

            // Update min and max for rho
            access_min_rho(0).update(rho);
            access_max_rho(0).update(rho);

            // Compute distance between GLL points in X direction
            if (ix < mesh.element_grid.ngll - 1) {
              auto current_point = global_coord_type(mesh.h_coord(0, ispec, ix, iz),
                                                     mesh.h_coord(1, ispec, ix, iz));
              auto next_point = global_coord_type(mesh.h_coord(0, ispec, ix + 1, iz),
                                                 mesh.h_coord(1, ispec, ix + 1, iz));
              type_real distance = specfem::point::distance(current_point,next_point);
              access_min_gll_distance(0).update(distance);
              access_max_gll_distance(0).update(distance);
              element_min_gll_distance = Kokkos::fmin(element_min_gll_distance, distance);
            }

            // Compute distance between GLL points in Z direction
            if (iz < mesh.element_grid.ngll - 1) {
              auto current_point = global_coord_type(mesh.h_coord(0, ispec, ix, iz),
                                                     mesh.h_coord(1, ispec, ix, iz));
              auto next_point = global_coord_type(mesh.h_coord(0, ispec, ix, iz + 1),
                                                 mesh.h_coord(1, ispec, ix, iz + 1));
              type_real distance = specfem::point::distance(current_point,next_point);
              access_min_gll_distance(0).update(distance);
              access_max_gll_distance(0).update(distance);
              element_min_gll_distance = Kokkos::fmin(element_min_gll_distance, distance);
            }
          }
        }

        // Element size assuming that ngllx = ngllz
        int fgll = mesh.element_grid.ngll - 1;
        for (int igll = 0; igll < mesh.element_grid.ngll; igll++) {

          // Z direction
          {
            auto current_point = global_coord_type(mesh.h_coord(0, ispec, 0, igll),
                                                   mesh.h_coord(1, ispec, 0, igll));
            auto next_point = global_coord_type(mesh.h_coord(0, ispec, fgll, igll),
                                                mesh.h_coord(1, ispec, fgll, igll));
            type_real distance = specfem::point::distance(current_point, next_point);
            access_min_distance(0).update(distance);
            access_max_distance(0).update(distance);
            element_max_distance = Kokkos::fmax(element_max_distance, distance);
          }

          // X direction
          {
            auto current_point = global_coord_type(mesh.h_coord(0, ispec, igll, 0),
                                                   mesh.h_coord(1, ispec, igll, 0));
            auto next_point = global_coord_type(mesh.h_coord(0, ispec, igll, fgll),
                                                mesh.h_coord(1, ispec, igll, fgll));
            type_real distance = specfem::point::distance(current_point, next_point);
            access_min_distance(0).update(distance);
            access_max_distance(0).update(distance);
            element_max_distance = Kokkos::fmax(element_max_distance, distance);
          }
        }

        // Compute average GLL distance from element size (using element-local max)
        type_real avg_distance = info::impl::compute_average_gll_spacing(element_max_distance, fgll);

        // Estimate largest minimum period resolved (using element-local min velocity)
        type_real max_minimum_period = info::impl::compute_minimum_period(avg_distance, element_min_v);

        // Update maximum minimum period
        access_max_minimum_period(0).update(max_minimum_period);

        // Suggested time step based on CFL condition (using element-local values)
        type_real element_dt_suggested = info::impl::compute_suggested_timestep(element_min_gll_distance, element_max_v);
        access_dt_suggested(0).update(element_dt_suggested);
      });
    };
  );
  
  // Finalize reductions
  Kokkos::Experimental::contribute(min_vp, scatter_vp_min);
  Kokkos::Experimental::contribute(max_vp, scatter_vp_max);
  Kokkos::Experimental::contribute(min_vs, scatter_vs_min);
  Kokkos::Experimental::contribute(max_vs, scatter_vs_max);
  Kokkos::Experimental::contribute(min_v, scatter_min_v);
  Kokkos::Experimental::contribute(max_v, scatter_max_v);
  Kokkos::Experimental::contribute(min_rho, scatter_rho_min);
  Kokkos::Experimental::contribute(max_rho, scatter_rho_max);
  Kokkos::Experimental::contribute(min_distance, scatter_distance_min);
  Kokkos::Experimental::contribute(max_distance, scatter_distance_max);
  Kokkos::Experimental::contribute(min_gll_distance, scatter_gll_distance_min);
  Kokkos::Experimental::contribute(max_gll_distance, scatter_gll_distance_max);
  Kokkos::Experimental::contribute(max_minimum_period, scatter_minimum_period_max);
  Kokkos::Experimental::contribute(dt_suggested, scatter_dt_suggested);
  Kokkos::fence();

  // Copy back to member variables
  auto min_vp_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), min_vp);
  auto max_vp_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), max_vp);
  this->vp = info::impl::Bounds(min_vp_h(0), max_vp_h(0));

  auto min_vs_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), min_vs);
  auto max_vs_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), max_vs);
  this->vs = info::impl::Bounds(min_vs_h(0), max_vs_h(0));

  auto min_v_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), min_v);
  auto max_v_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), max_v);
  this->v = info::impl::Bounds(min_v_h(0), max_v_h(0)); 
  auto min_rho_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), min_rho);
  auto max_rho_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), max_rho);
  this->rho = info::impl::Bounds(min_rho_h(0), max_rho_h(0));  

  auto min_distance_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), min_distance);
  auto max_distance_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), max_distance);
  this->element_size = info::impl::Bounds(min_distance_h(0), max_distance_h(0)); 
  auto min_gll_distance_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), min_gll_distance);
  auto max_gll_distance_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), max_gll_distance);
  this->gll_distance = info::impl::Bounds(min_gll_distance_h(0), max_gll_distance_h(0));

  auto max_minimum_period_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), max_minimum_period);
  this->largest_minimum_period = max_minimum_period_h(0);

  auto dt_suggested_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), dt_suggested);
  this->suggested_time_step = dt_suggested_h(0);
  
  // Largest minimum period resolved
  this->largest_minimum_period = max_minimum_period_h(0);

}