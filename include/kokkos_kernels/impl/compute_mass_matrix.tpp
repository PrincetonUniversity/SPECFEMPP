#pragma once

#include "boundary_conditions/boundary_conditions.hpp"
#include "boundary_conditions/boundary_conditions.tpp"
#include "datatypes/simd.hpp"
#include "element/quadrature.hpp"
#include "enumerations/dimension.hpp"
#include "enumerations/medium.hpp"
#include "enumerations/wavefield.hpp"
#include "execution/chunked_domain_iterator.hpp"
#include "execution/for_all.hpp"
#include "medium/compute_mass_matrix.hpp"
#include "parallel_configuration/chunk_config.hpp"
#include "specfem/assembly.hpp"
#include "specfem/point.hpp"
#include <Kokkos_Core.hpp>

template <specfem::dimension::type DimensionTag,
          specfem::wavefield::simulation_field WavefieldType, int NGLL,
          specfem::element::medium_tag MediumTag,
          specfem::element::property_tag PropertyTag,
          specfem::element::boundary_tag BoundaryTag>
void specfem::kokkos_kernels::impl::compute_mass_matrix(
    const type_real &dt,
    const specfem::assembly::assembly<DimensionTag> &assembly) {

  constexpr auto dimension = DimensionTag;
  constexpr auto wavefield = WavefieldType;
  constexpr auto medium_tag = MediumTag;
  constexpr auto property_tag = PropertyTag;
  constexpr auto boundary_tag = BoundaryTag;

  // Get the number of components for the element
  constexpr int components =
      specfem::element::attributes<dimension, medium_tag>::components;

  const auto elements = assembly.element_types.get_elements_on_device(
      MediumTag, PropertyTag, BoundaryTag);

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
    throw std::runtime_error("The number of GLL points in the mesh elements must match "
                             "the template parameter NGLL.");
  }

#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
  constexpr bool using_simd = false;
#else
  constexpr bool using_simd = true;
#endif
  using simd = specfem::datatype::simd<type_real, using_simd>;
  using parallel_config = specfem::parallel_config::default_chunk_config<
      dimension, simd, Kokkos::DefaultExecutionSpace>;

  using PointMassType =
      specfem::point::mass_inverse<dimension, medium_tag, using_simd>;

  using PointPropertyType =
      specfem::point::properties<dimension, medium_tag, property_tag,
                                 using_simd>;

  using PointJacobianMatrixType =
      specfem::point::jacobian_matrix<dimension, true, using_simd>;

  using PointBoundaryType =
      specfem::point::boundary<boundary_tag, dimension, using_simd>;

  using PointIndex = specfem::point::index<dimension, using_simd>;

  const auto &mesh = assembly.mesh;
  const auto &jacobian_matrix = assembly.jacobian_matrix;
  const auto &properties = assembly.properties;
  const auto &boundaries = assembly.boundaries;
  const auto field = assembly.fields.template get_simulation_field<wavefield>();

  const auto wgll = mesh.weights;

  specfem::execution::ChunkedDomainIterator chunk(parallel_config(), elements,
                                                  element_grid);

  specfem::execution::for_all(
      "specfem::kokkos_kernels::compute_mass_matrix", chunk,
      KOKKOS_LAMBDA(const PointIndex &index) {
        const int ix = index.ix;
        const int iz = index.iz;

        const auto point_property = [&]() -> PointPropertyType {
          PointPropertyType point_property;

          specfem::assembly::load_on_device(index, properties, point_property);
          return point_property;
        }();

        const auto jacobian = [&]() {
          PointJacobianMatrixType point_jacobian_matrix;
          specfem::assembly::load_on_device(index, jacobian_matrix,
                                            point_jacobian_matrix);
          return point_jacobian_matrix.jacobian;
        }();

        PointMassType mass_matrix =
            specfem::medium::mass_matrix_component(point_property);

        mass_matrix *= wgll(ix) * wgll(iz) * jacobian;

        PointBoundaryType point_boundary;
        specfem::assembly::load_on_device(index, boundaries, point_boundary);

        specfem::boundary_conditions::compute_mass_matrix_terms(
            dt, point_boundary, point_property, mass_matrix);

        specfem::assembly::atomic_add_on_device(index, field, mass_matrix);
      });
}
