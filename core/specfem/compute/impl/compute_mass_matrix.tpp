#pragma once

#include "compute_mass_matrix.hpp"
#include "specfem/assembly/assembly.hpp"
#include "specfem/boundary_conditions.hpp"
#include "specfem/datatype.hpp"
#include "specfem/element.hpp"
#include "specfem/enums.hpp"
#include "specfem/execution.hpp"
#include "specfem/medium_physics.hpp"
#include "specfem/parallel_configuration.hpp"
#include "specfem/point.hpp"
#include "specfem/quadrature.hpp"
#include <Kokkos_Core.hpp>

template <int NGLL, typename Tags>
void specfem::compute::impl::compute_mass_matrix(
    const type_real &dt,
    const specfem::assembly::assembly<Tags::dimension_tag> &assembly) {

  constexpr auto dimension_tag = Tags::dimension_tag;
  constexpr auto wavefield = Tags::wavefield_tag;
  constexpr auto medium_tag = Tags::medium_tag;
  constexpr auto property_tag = Tags::property_tag;
  constexpr auto boundary_tag = Tags::boundary_tag;
  // `attenuation_tag` is always supplied: the public `compute_mass_matrix`
  // dispatcher iterates an ATTENUATION_SET, and the mpi-tag-aware element
  // accessors below are keyed by attenuation. A direct caller whose Tags lack
  // an attenuation tag will fail this access at compile time, by design.
  constexpr auto attenuation_tag = Tags::attenuation_tag;

  const auto elements = assembly.element_types.get_elements_on_device(
          medium_tag, property_tag, attenuation_tag, boundary_tag,
          Tags::mpi_tag);

  // Get number of elements matching the tag combinations
  const int nelements = elements.extent(0);

  // Return if no elements match the tag combination
  if (nelements == 0)
    return;

  // Get the element grid (ngllx, ngllz)
  const auto &element_grid = assembly.mesh.element_grid;

  // Check if the number of GLL points in the mesh elements matches the template
  // parameter NGLL
  if (element_grid != NGLL) {
    throw std::runtime_error(
        "The number of GLL points in the mesh elements must match "
        "the template parameter NGLL.");
  }

#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
  constexpr bool using_simd = false;
#else
  constexpr bool using_simd = true;
#endif

  using simd = specfem::datatype::simd<type_real, using_simd>;
  using parallel_config = specfem::parallel_configuration::default_chunk_config<
      dimension_tag, simd, Kokkos::DefaultExecutionSpace>;

  using PointTags =
      specfem::tags::Tags<dimension_tag, medium_tag, property_tag, using_simd>;

  using PointMassType = specfem::point::mass_inverse<PointTags>;

  using PointPropertyType = specfem::point::properties<PointTags>;

  using PointJacobianMatrixType =
      specfem::point::jacobian_matrix<dimension_tag, true, using_simd>;

  using PointBoundaryType =
      specfem::point::boundary<boundary_tag, dimension_tag, using_simd>;

  using PointWeightsType = specfem::point::weights<dimension_tag>;

  using PointIndex = specfem::point::index<dimension_tag, using_simd>;

  const auto &mesh = assembly.mesh;
  const auto &jacobian_matrix = assembly.jacobian_matrix;
  const auto &properties = assembly.properties;
  const auto &boundaries = assembly.boundaries;
  const auto field = assembly.fields.template get_simulation_field<wavefield>();

  specfem::execution::ChunkedDomainIterator chunk(parallel_config(), elements,
                                                  element_grid);

  specfem::execution::for_all(
      "specfem::compute::compute_mass_matrix", chunk,
      KOKKOS_LAMBDA(
          const typename decltype(chunk)::base_index_type &iterator_index) {
        const auto index = iterator_index.get_index();
        PointPropertyType point_property;
        specfem::assembly::load_on_device(index, properties, point_property);

        PointJacobianMatrixType point_jacobian_matrix;
        specfem::assembly::load_on_device(index, jacobian_matrix,
                                          point_jacobian_matrix);

        PointWeightsType point_weights;
        specfem::assembly::load_on_device(index, mesh.weights, point_weights);

        PointMassType mass_matrix =
            specfem::medium_physics::mass_matrix_component<PointTags>(
                point_property);

        mass_matrix *= point_weights.product() * point_jacobian_matrix.jacobian;

        PointBoundaryType point_boundary;
        specfem::assembly::load_on_device(index, boundaries, point_boundary);

        specfem::boundary_conditions::compute_mass_matrix_terms(
            dt, point_boundary, point_property, mass_matrix);

        specfem::assembly::atomic_add_on_device(index, field, mass_matrix);
      });
}
