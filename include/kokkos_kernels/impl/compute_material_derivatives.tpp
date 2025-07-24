#pragma once

#include "algorithms/gradient.hpp"
#include "chunk_element/field.hpp"
#include "compute_material_derivatives.hpp"
#include "execution/chunked_domain_iterator.hpp"
#include "execution/for_each_level.hpp"
#include "medium/compute_frechet_derivatives.hpp"
#include "parallel_configuration/chunk_config.hpp"
#include "specfem/assembly.hpp"
#include "specfem/point.hpp"
#include <Kokkos_Core.hpp>

template <specfem::dimension::type DimensionTag, int NGLL,
          specfem::element::medium_tag MediumTag,
          specfem::element::property_tag PropertyTag>
void specfem::kokkos_kernels::impl::compute_material_derivatives(
    const specfem::assembly::assembly<DimensionTag> &assembly,
    const type_real &dt) {
  auto &properties = assembly.properties;
  auto &kernels = assembly.kernels;
  auto &adjoint_field = assembly.fields.adjoint;
  auto &backward_field = assembly.fields.backward;
  auto &mesh = assembly.mesh;
  auto &jacobian_matrix = assembly.jacobian_matrix;

  const auto elements =
      assembly.element_types.get_elements_on_device(MediumTag, PropertyTag);

  const int nelements = elements.extent(0);

  const int ngllz = mesh.ngllz;
  const int ngllx = mesh.ngllx;

  if (nelements == 0) {
    return;
  }

#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
  constexpr bool using_simd = false;
#else
  constexpr bool using_simd = true;
#endif

  if (ngllz != NGLL || ngllx != NGLL) {
    throw std::runtime_error(
        "The number of GLL points in z and x must match the template parameter "
        "NGLL.");
  }

  using simd = specfem::datatype::simd<type_real, using_simd>;
  using ParallelConfig = specfem::parallel_config::default_chunk_config<
      DimensionTag, simd, Kokkos::DefaultExecutionSpace>;

  using ChunkElementFieldType = specfem::chunk_element::field<
      ParallelConfig::chunk_size, NGLL, DimensionTag, MediumTag,
      specfem::kokkos::DevScratchSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>,
      true, false, false, false, using_simd>;

  using ElementQuadratureType = specfem::element::quadrature<
      NGLL, DimensionTag, specfem::kokkos::DevScratchSpace,
      Kokkos::MemoryTraits<Kokkos::Unmanaged>, true, false>;

  using PointDisplacementType =
      specfem::point::displacement<DimensionTag, MediumTag, using_simd>;
  using PointVelocityType =
      specfem::point::velocity<DimensionTag, MediumTag, using_simd>;
  using PointAccelerationType =
      specfem::point::acceleration<DimensionTag, MediumTag, using_simd>;

  using PointFieldDerivativesType =
      specfem::point::field_derivatives<DimensionTag, MediumTag, using_simd>;

  using PointPropertiesType =
      specfem::point::properties<DimensionTag, MediumTag, PropertyTag,
                                 using_simd>;

  int scratch_size = 2 * ChunkElementFieldType::shmem_size() +
                     ElementQuadratureType::shmem_size();

  specfem::execution::ChunkedDomainIterator chunk(ParallelConfig(), elements,
                                                  ngllz, ngllx);

  specfem::execution::for_each_level(
      "specfem::kokkos_kernels::compute_material_derivatives",
      chunk.set_scratch_size(0, Kokkos::PerTeam(scratch_size)),
      KOKKOS_LAMBDA(const typename decltype(chunk)::index_type &chunk_index) {
        const auto team = chunk_index.get_policy_index();
        ChunkElementFieldType adjoint_element_field(team);
        ChunkElementFieldType backward_element_field(team);
        ElementQuadratureType quadrature_element(team);
        specfem::assembly::load_on_device(team, mesh, quadrature_element);

        // Load the element index
        specfem::assembly::load_on_device(chunk_index, adjoint_field,
                                          adjoint_element_field);
        specfem::assembly::load_on_device(chunk_index, backward_field,
                                          backward_element_field);
        team.team_barrier();

        // Generate the Kernels
        // We call the gradient algorith, which computes the gradient of
        // adjoint and backward fields at each point in the element
        // The Lambda function is is passed to the gradient algorithm
        // which is applied to gradient result for every quadrature point

        specfem::algorithms::gradient(
            chunk_index, jacobian_matrix, quadrature_element.hprime_gll,
            adjoint_element_field.displacement,
            backward_element_field.displacement,
            [&](const auto &iterator_index,
                const typename PointFieldDerivativesType::value_type &df,
                const typename PointFieldDerivativesType::value_type &dg) {
              const auto index = iterator_index.get_index();
              // Load properties, adjoint field, and backward field
              // for the point
              // ------------------------------
              const auto point_properties = [&]() -> PointPropertiesType {
                PointPropertiesType point_properties;
                specfem::assembly::load_on_device(index, properties,
                                                  point_properties);
                return point_properties;
              }();

              PointVelocityType adjoint_velocity;
              PointAccelerationType adjoint_acceleration;
              specfem::assembly::load_on_device(
                  index, adjoint_field, adjoint_velocity, adjoint_acceleration);

              PointDisplacementType backward_displacement;
              specfem::assembly::load_on_device(index, backward_field,
                                                backward_displacement);
              // ------------------------------

              const PointFieldDerivativesType adjoint_point_derivatives(df);
              const PointFieldDerivativesType backward_point_derivatives(dg);

              // Compute the kernel for the point
              const auto point_kernel =
                  specfem::medium::compute_frechet_derivatives(
                      point_properties, adjoint_velocity, adjoint_acceleration,
                      backward_displacement, adjoint_point_derivatives,
                      backward_point_derivatives, dt);

              // Update the kernel in the global memory
              specfem::assembly::add_on_device(index, point_kernel, kernels);
            });
      });
}
