#pragma once

#include "boundary_conditions/boundary_conditions.hpp"
#include "boundary_conditions/boundary_conditions.tpp"
#include "compute/assembly/assembly.hpp"
#include "datatypes/simd.hpp"
#include "element/quadrature.hpp"
#include "enumerations/dimension.hpp"
#include "enumerations/medium.hpp"
#include "enumerations/wavefield.hpp"
#include "medium/compute_mass_matrix.hpp"
#include "parallel_configuration/chunk_config.hpp"
#include "execution/chunked_domain_iterator.hpp"
#include "execution/for_all.hpp"
#include "specfem/point.hpp"
#include <Kokkos_Core.hpp>

template <specfem::dimension::type DimensionTag,
          specfem::wavefield::simulation_field WavefieldType, int NGLL,
          specfem::element::medium_tag MediumTag,
          specfem::element::property_tag PropertyTag,
          specfem::element::boundary_tag BoundaryTag>
void specfem::kokkos_kernels::impl::compute_mass_matrix(
    const type_real &dt, const specfem::compute::assembly &assembly) {

  constexpr auto dimension = DimensionTag;
  constexpr auto wavefield = WavefieldType;
  constexpr auto medium_tag = MediumTag;
  constexpr auto property_tag = PropertyTag;
  constexpr auto boundary_tag = BoundaryTag;

  const auto elements = assembly.element_types.get_elements_on_device(
      MediumTag, PropertyTag, BoundaryTag);

  constexpr int components =
      specfem::element::attributes<dimension, medium_tag>::components;

  const int ngllz = assembly.mesh.ngllz;
  const int ngllx = assembly.mesh.ngllx;

  const int nelements = elements.extent(0);

  if (nelements == 0)
    return;

#ifdef KOKKOS_ENABLE_CUDA
  constexpr bool using_simd = false;
#else
  constexpr bool using_simd = true;
#endif
  using simd = specfem::datatype::simd<type_real, using_simd>;
  using parallel_config = specfem::parallel_config::default_chunk_config<
      dimension, simd, Kokkos::DefaultExecutionSpace>;

  using PointMassType = specfem::point::field<dimension, medium_tag, false,
                                              false, false, true, using_simd>;

  using PointPropertyType =
      specfem::point::properties<dimension, medium_tag, property_tag,
                                 using_simd>;

  using PointPartialDerivativesType =
      specfem::point::partial_derivatives<dimension, true, using_simd>;

  using PointBoundaryType =
      specfem::point::boundary<boundary_tag, dimension, using_simd>;

  using PointIndex = specfem::point::index<dimension, using_simd>;

  const auto &quadrature = assembly.mesh.quadratures;
  const auto &partial_derivatives = assembly.partial_derivatives;
  const auto &properties = assembly.properties;
  const auto &boundaries = assembly.boundaries;
  const auto field = assembly.fields.get_simulation_field<wavefield>();

  const auto wgll = quadrature.gll.weights;

  specfem::execution::ChunkedDomainIterator chunk(parallel_config(), elements, ngllz, ngllx);

  specfem::execution::for_all(
      "specfem::kokkos_kernels::compute_mass_matrix", chunk,
      KOKKOS_LAMBDA(const PointIndex &index) {
        const int ix = index.ix;
        const int iz = index.iz;

        const auto point_property = [&]() -> PointPropertyType {
          PointPropertyType point_property;

          specfem::compute::load_on_device(index, properties, point_property);
          return point_property;
        }();

        const auto jacobian = [&]() {
          PointPartialDerivativesType point_partial_derivatives;
          specfem::compute::load_on_device(index, partial_derivatives,
                                           point_partial_derivatives);
          return point_partial_derivatives.jacobian;
        }();

        PointMassType mass_matrix =
            specfem::medium::mass_matrix_component(point_property);

        for (int icomp = 0; icomp < components; icomp++) {
          mass_matrix.mass_matrix(icomp) *= wgll(ix) * wgll(iz) * jacobian;
        }

        PointBoundaryType point_boundary;
        specfem::compute::load_on_device(index, boundaries, point_boundary);

        specfem::boundary_conditions::compute_mass_matrix_terms(
            dt, point_boundary, point_property, mass_matrix);

        specfem::compute::atomic_add_on_device(index, mass_matrix, field);
      });
}
