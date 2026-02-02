#include "info.hpp"
#include "specfem/assembly/assembly.hpp"
#include "specfem/assembly/info/impl/bounds.hpp"
#include "specfem/assembly/info/impl/compute.hpp"
#include "specfem/assembly/info/impl/scatter_minmax.hpp"
#include "specfem/point.hpp"
#include "specfem_setup.hpp"
#include "impl/bounding_box.hpp"
#include <limits>

specfem::assembly::Info<specfem::dimension::type::dim2>::Info(
    specfem::assembly::assembly<dimension_tag> &assembly) {

  // Get reference to element types, and mesh
  const auto &element_types = assembly.element_types;
  const auto &mesh = assembly.mesh;

  // Domain bounds
  domain_bounds = info::impl::BoundingBox<dimension_tag>(mesh.xmin, mesh.xmax,
                                                         mesh.zmin, mesh.zmax);

  // Create scatter min/max reducers
  using info::impl::ScatterMinMax;
  ScatterMinMax<type_real> vp_scatter("vp");
  ScatterMinMax<type_real> vs_scatter("vs");
  ScatterMinMax<type_real> v_scatter("v");
  ScatterMinMax<type_real> rho_scatter("rho");
  ScatterMinMax<type_real> distance_scatter("distance");
  ScatterMinMax<type_real> gll_distance_scatter("gll_distance");
  ScatterMinMax<type_real> minimum_period_scatter("minimum_period");
  ScatterMinMax<type_real> dt_scatter("dt");

  // Type alias
  using global_coord_type =
      specfem::point::global_coordinates<specfem::dimension::type::dim2>;
  using info::impl::LocalMinMax;

  FOR_EACH_IN_PRODUCT(
      (DIMENSION_TAG(DIM2),
       MEDIUM_TAG(ELASTIC_PSV, ELASTIC_SH, ACOUSTIC, POROELASTIC,
                  ELASTIC_PSV_T),
       PROPERTY_TAG(ISOTROPIC, ANISOTROPIC, ISOTROPIC_COSSERAT)),
      {
        const auto dimension_tag = _dimension_tag_;
        const auto medium_tag = _medium_tag_;
        const auto property_tag = _property_tag_;

        auto elements =
            element_types.get_elements_on_host(medium_tag, property_tag);

        // Compute min and max using scatter reduce
        Kokkos::parallel_for(
            "compute_minmax", elements.extent(0), KOKKOS_LAMBDA(int i) {
              auto ispec = elements(i);

              // Get scatter accessors
              auto vp_acc = vp_scatter.access();
              auto vs_acc = vs_scatter.access();
              auto v_acc = v_scatter.access();
              auto rho_acc = rho_scatter.access();
              auto distance_acc = distance_scatter.access();
              auto gll_distance_acc = gll_distance_scatter.access();
              auto minimum_period_acc = minimum_period_scatter.access();
              auto dt_acc = dt_scatter.access();

              // Create point property object
              specfem::point::properties<dimension_tag, medium_tag,
                                         property_tag, false>
                  point_property;

              // Element-local tracking for per-element calculations
              LocalMinMax<type_real> element_v;
              LocalMinMax<type_real> element_gll_distance;
              LocalMinMax<type_real> element_distance;

              // Get the material properties for this element
              for (int iz = 0; iz < mesh.element_grid.ngll; iz++) {
                for (int ix = 0; ix < mesh.element_grid.ngll; ix++) {

                  // Create point index
                  specfem::point::index<dimension_tag> point_index(ispec, ix,
                                                                   iz);

                  // Load point properties
                  specfem::assembly::load_on_host(point_index,
                                                  assembly.properties,
                                                  point_property);

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

                  // Update element-local velocity bounds
                  element_v.update_min(vmin);
                  element_v.update_max(vmax);

                  // Compute distance between GLL points in X direction
                  if (ix < mesh.element_grid.ngll - 1) {
                    auto current_point = global_coord_type(
                        mesh.h_coord(0, ispec, ix, iz),
                        mesh.h_coord(1, ispec, ix, iz));
                    auto next_point = global_coord_type(
                        mesh.h_coord(0, ispec, ix + 1, iz),
                        mesh.h_coord(1, ispec, ix + 1, iz));
                    type_real dist =
                        specfem::point::distance(current_point, next_point);
                    gll_distance_acc.update(dist);
                    element_gll_distance.update(dist);
                  }

                  // Compute distance between GLL points in Z direction
                  if (iz < mesh.element_grid.ngll - 1) {
                    auto current_point = global_coord_type(
                        mesh.h_coord(0, ispec, ix, iz),
                        mesh.h_coord(1, ispec, ix, iz));
                    auto next_point = global_coord_type(
                        mesh.h_coord(0, ispec, ix, iz + 1),
                        mesh.h_coord(1, ispec, ix, iz + 1));
                    type_real dist =
                        specfem::point::distance(current_point, next_point);
                    gll_distance_acc.update(dist);
                    element_gll_distance.update(dist);
                  }
                }
              }

              // Element size assuming that ngllx = ngllz
              int fgll = mesh.element_grid.ngll - 1;
              for (int igll = 0; igll < mesh.element_grid.ngll; igll++) {

                // Z direction
                {
                  auto current_point =
                      global_coord_type(mesh.h_coord(0, ispec, 0, igll),
                                        mesh.h_coord(1, ispec, 0, igll));
                  auto next_point =
                      global_coord_type(mesh.h_coord(0, ispec, fgll, igll),
                                        mesh.h_coord(1, ispec, fgll, igll));
                  type_real dist =
                      specfem::point::distance(current_point, next_point);
                  distance_acc.update(dist);
                  element_distance.update(dist);
                }

                // X direction
                {
                  auto current_point =
                      global_coord_type(mesh.h_coord(0, ispec, igll, 0),
                                        mesh.h_coord(1, ispec, igll, 0));
                  auto next_point =
                      global_coord_type(mesh.h_coord(0, ispec, igll, fgll),
                                        mesh.h_coord(1, ispec, igll, fgll));
                  type_real dist =
                      specfem::point::distance(current_point, next_point);
                  distance_acc.update(dist);
                  element_distance.update(dist);
                }
              }

              // Compute average GLL distance from element size
              type_real avg_distance =
                  info::impl::compute_average_gll_spacing(
                      element_distance.max_val, fgll);

              // Estimate minimum period resolved (using element-local min
              // velocity)
              type_real min_period = info::impl::compute_minimum_period(
                  avg_distance, element_v.min_val);

              // Update minimum period bounds
              minimum_period_acc.update(min_period);

              // Suggested time step based on CFL condition
              type_real element_dt = info::impl::compute_suggested_timestep(
                  element_gll_distance.min_val, element_v.max_val);
              dt_acc.update(element_dt);
            });
      };);

  // Finalize reductions
  vp_scatter.contribute();
  vs_scatter.contribute();
  v_scatter.contribute();
  rho_scatter.contribute();
  distance_scatter.contribute();
  gll_distance_scatter.contribute();
  minimum_period_scatter.contribute();
  dt_scatter.contribute();
  Kokkos::fence();

  // Copy results to member variables
  this->vp = vp_scatter.get_bounds();
  this->vs = vs_scatter.get_bounds();
  this->v = v_scatter.get_bounds();
  this->rho = rho_scatter.get_bounds();
  this->element_size = distance_scatter.get_bounds();
  this->gll_distance = gll_distance_scatter.get_bounds();

  auto period_bounds = minimum_period_scatter.get_bounds();
  this->largest_minimum_period = period_bounds.max;

  auto dt_bounds = dt_scatter.get_bounds();
  this->suggested_time_step = dt_bounds.min;
}
