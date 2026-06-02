#pragma once

#include "specfem/algorithms.hpp"
#include "specfem/assembly/assembly.hpp"
#include "specfem/boundary_conditions.hpp"
#include "specfem/chunk_element.hpp"
#include "specfem/datatype.hpp"
#include "specfem/element.hpp"
#include "specfem/enums.hpp"
#include "specfem/execution.hpp"
#include "specfem/macros.hpp"
#include "specfem/medium_physics.hpp"
#include "specfem/parallel_configuration.hpp"
#include "specfem/point.hpp"
#include "specfem/quadrature.hpp"
#include "specfem/tags.hpp"
#include <Kokkos_Core.hpp>
#include <type_traits>

namespace specfem::compute::impl {

// ---------------------------------------------------------------------------
// Gather kernel: two-pass (gradient → store F, divergence → accumulate accel)
// Uses more shared memory than scatter but avoids atomic operations.
// ---------------------------------------------------------------------------

template <int NGLL, typename Tags> class gather_kernel {
public:
  constexpr static auto medium_tag = Tags::medium_tag;
  constexpr static auto property_tag = Tags::property_tag;
  constexpr static auto boundary_tag = Tags::boundary_tag;
  constexpr static auto attenuation_tag = Tags::attenuation_tag;
  constexpr static auto dimension_tag = Tags::dimension_tag;
  constexpr static int ngll = NGLL;

#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
  constexpr static bool using_simd = false;
#else
  // TODO(Rohit : DIM3_SIMD) Enable simd execution for dim3 solver
  constexpr static bool using_simd =
      (Tags::dimension_tag == specfem::element::dimension_tag::dim2) ? true
                                                                     : false;
#endif

  using simd = specfem::datatype::simd<type_real, using_simd>;

  using ParallelConfig = specfem::parallel_configuration::default_chunk_config<
      Tags::dimension_tag, simd, Kokkos::DefaultExecutionSpace>;

  using ChunkFieldPackType = specfem::chunk_element::stiffness_field_pack<
      attenuation_tag, ParallelConfig::chunk_size, ngll, dimension_tag,
      medium_tag, using_simd>;

  using ChunkStressIntegrandType = specfem::chunk_element::stress_integrand<
      ParallelConfig::chunk_size, ngll, Tags::dimension_tag, Tags::medium_tag,
      Kokkos::DefaultExecutionSpace::scratch_memory_space,
      Kokkos::MemoryTraits<Kokkos::Unmanaged>, using_simd>;

  using ElementQuadratureType = specfem::quadrature::lagrange_derivative<
      ngll, Tags::dimension_tag,
      Kokkos::DefaultExecutionSpace::scratch_memory_space,
      Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

  using PointTags = specfem::tags::Tags<Tags::dimension_tag, Tags::medium_tag,
                                        Tags::property_tag,
                                        Tags::attenuation_tag, using_simd>;

  using PointBoundaryType =
      specfem::point::boundary<Tags::boundary_tag, Tags::dimension_tag,
                               using_simd>;
  using PointDisplacementType = specfem::point::displacement<PointTags>;
  using PointVelocityType = specfem::point::velocity<PointTags>;
  using PointAccelerationType = specfem::point::acceleration<PointTags>;
  using PointJacobianMatrixType =
      specfem::point::jacobian_matrix<Tags::dimension_tag, true, using_simd>;
  using PointPropertyType = specfem::point::properties<PointTags>;
  using PointFieldDerivativesType =
      specfem::point::field_derivatives<PointTags>;
  using PointStressType = specfem::point::stress<PointTags>;

  using GradientPackType = specfem::point::stiffness_gradient_pack<
      attenuation_tag, typename PointFieldDerivativesType::value_type>;

  using PointWeightsType = specfem::point::weights<Tags::dimension_tag>;

  constexpr static int ncomp =
      specfem::element::attributes<dimension_tag, medium_tag>::components;

  constexpr static bool has_cosserat_couple_stress =
      specfem::element::attributes<dimension_tag,
                                   medium_tag>::has_cosserat_couple_stress;

  using AssemblyType = specfem::assembly::assembly<Tags::dimension_tag>;

  gather_kernel(const AssemblyType &assembly, int istep)
      : assembly_(assembly), istep_(istep) {}

  static int shmem_size() {
    return ChunkFieldPackType::shmem_size() +
           ChunkStressIntegrandType::shmem_size() +
           ElementQuadratureType::shmem_size();
  }

  template <typename ChunkType> void operator()(ChunkType &chunk) const {
    const auto &mesh = assembly_.mesh;
    const auto &jacobian_matrix = assembly_.jacobian_matrix;
    const auto &properties = assembly_.properties;
    const auto &boundaries = assembly_.boundaries;
    const auto &attenuation = assembly_.attenuation;
    const auto &field =
        assembly_.fields.template get_simulation_field<Tags::wavefield_tag>();
    const auto &boundary_values =
        assembly_.boundary_values.template get_container<boundary_tag>();
    const int istep = istep_;

    specfem::execution::for_each_level(
        "specfem::compute::compute_stiffness_interaction::gather",
        chunk.set_scratch_size(0, Kokkos::PerTeam(shmem_size())),
        KOKKOS_LAMBDA(
            const typename ChunkType::index_type &chunk_iterator_index) {
          const auto &chunk_index = chunk_iterator_index.get_index();
          const auto team = chunk_index.get_policy_index();

          ChunkFieldPackType field_pack(team.team_scratch(0));
          ElementQuadratureType lagrange_derivative(team);
          ChunkStressIntegrandType stress_integrand(team);

          specfem::assembly::load_on_device(team, mesh, lagrange_derivative);
          specfem::assembly::load_on_device(chunk_index, field, field_pack);

          team.team_barrier();

          specfem::algorithms::gradient(
              chunk_index, jacobian_matrix, lagrange_derivative, field_pack,
              [&](const auto &iterator_index,
                  const GradientPackType &grad_pack) {
                const auto &index = iterator_index.get_index();
                const auto &local_index = iterator_index.get_local_index();

                PointJacobianMatrixType point_jacobian_matrix;
                specfem::assembly::load_on_device(index, jacobian_matrix,
                                                  point_jacobian_matrix);

                PointPropertyType point_property;
                specfem::assembly::load_on_device(index, properties,
                                                  point_property);

                const auto field_derivatives =
                    grad_pack.template get_du<PointTags>();

                const auto field_derivatives_velocity =
                    grad_pack.template get_dv<PointTags>();

                PointDisplacementType point_displacement;
                specfem::assembly::load_on_device(index, field,
                                                  point_displacement);

                auto point_stress =
                    specfem::medium_physics::compute_stress<PointTags>(
                        point_property, field_derivatives);

                specfem::medium_physics::compute_attenuation<PointTags>(
                    index, point_stress, grad_pack, attenuation);

                specfem::medium_physics::compute_cosserat_stress(
                    point_property, point_displacement, point_stress);

                stress_integrand.F(local_index) =
                    point_stress * point_jacobian_matrix;
              });

          team.team_barrier();

          specfem::algorithms::divergence(
              chunk_index, mesh.weights, lagrange_derivative,
              stress_integrand.F,
              [&](const auto &iterator_index,
                  const typename PointAccelerationType::value_type &result) {
                const auto &index = iterator_index.get_index();
                const auto &local_index = iterator_index.get_local_index();

                PointAccelerationType acceleration(result);
                acceleration *= static_cast<type_real>(-1.0);

                PointPropertyType point_property;
                specfem::assembly::load_on_device(index, properties,
                                                  point_property);

                PointVelocityType velocity;
                specfem::assembly::load_on_device(index, field, velocity);

                PointBoundaryType point_boundary;
                specfem::assembly::load_on_device(index, boundaries,
                                                  point_boundary);

                PointWeightsType point_weights;
                specfem::assembly::load_on_device(index, mesh.weights,
                                                  point_weights);

                PointJacobianMatrixType point_jacobian_matrix;
                specfem::assembly::load_on_device(index, jacobian_matrix,
                                                  point_jacobian_matrix);

                const auto factor =
                    point_weights.product() * point_jacobian_matrix.jacobian;

                specfem::medium_physics::compute_damping_force<decltype(factor),
                                                               PointTags>(
                    factor, point_property, velocity, acceleration);

                // Recover physical stress T from stored stress integrand F.
                // F[i][k] = J*(T[i][0]*xix + T[i][1]*xiz) for k=0,
                //            J*(T[i][0]*gammax + T[i][1]*gammaz) for k=1.
                // Inverting: T[i][0] = F[i][0]*gammaz - F[i][1]*xiz
                //            T[i][1] = -F[i][0]*gammax + F[i][1]*xix
                if constexpr (has_cosserat_couple_stress) {
                  PointStressType T_recovered;
                  for (int icomp = 0; icomp < ncomp; ++icomp) {
                    T_recovered.T(icomp, 0) =
                        stress_integrand.F(local_index, icomp, 0) *
                            point_jacobian_matrix.gammaz -
                        stress_integrand.F(local_index, icomp, 1) *
                            point_jacobian_matrix.xiz;
                    T_recovered.T(icomp, 1) =
                        -stress_integrand.F(local_index, icomp, 0) *
                            point_jacobian_matrix.gammax +
                        stress_integrand.F(local_index, icomp, 1) *
                            point_jacobian_matrix.xix;
                  }
                  // acceleration is already negated;
                  // compute_cosserat_couple_stress accumulates positively, so
                  // subtract delta to stay consistent.
                  PointAccelerationType cosserat_delta;
                  for (int icomp = 0; icomp < ncomp; ++icomp)
                    cosserat_delta(icomp) = static_cast<type_real>(0);
                  specfem::medium_physics::compute_cosserat_couple_stress(
                      point_property, factor, T_recovered, cosserat_delta);
                  for (int icomp = 0; icomp < ncomp; ++icomp)
                    acceleration(icomp) -= cosserat_delta(icomp);
                }

                specfem::boundary_conditions::apply_boundary_conditions(
                    point_boundary, point_property, velocity, acceleration);

                if (Tags::wavefield_tag ==
                    specfem::simulation::field_type::forward) {
                  specfem::assembly::store_on_device(istep, index, acceleration,
                                                     boundary_values);
                }

                specfem::assembly::atomic_add_on_device(index, field,
                                                        acceleration);
              });
        });
  }

private:
  const AssemblyType &assembly_;
  int istep_;
};

// ---------------------------------------------------------------------------
// Scatter kernel: single-pass with atomic accumulation into acceleration
// scratch. Uses less shared memory than gather but has atomic operation
// overhead.
// ---------------------------------------------------------------------------

template <int NGLL, typename Tags> class scatter_kernel {
public:
  constexpr static auto medium_tag = Tags::medium_tag;
  constexpr static auto property_tag = Tags::property_tag;
  constexpr static auto boundary_tag = Tags::boundary_tag;
  constexpr static auto attenuation_tag = Tags::attenuation_tag;
  constexpr static auto dimension_tag = Tags::dimension_tag;
  constexpr static int ngll = NGLL;

#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
  constexpr static bool using_simd = false;
#else
  constexpr static bool using_simd =
      (Tags::dimension_tag == specfem::element::dimension_tag::dim2) ? true
                                                                     : false;
#endif

  using simd = specfem::datatype::simd<type_real, using_simd>;

  using ParallelConfig = specfem::parallel_configuration::default_chunk_config<
      Tags::dimension_tag, simd, Kokkos::DefaultExecutionSpace>;

  using ChunkFieldPackType = specfem::chunk_element::stiffness_field_pack<
      attenuation_tag, ParallelConfig::chunk_size, ngll, dimension_tag,
      medium_tag, using_simd>;

  using ChunkAccelerationType = specfem::chunk_element::acceleration<
      ParallelConfig::chunk_size, ngll, dimension_tag, medium_tag, using_simd>;

  using ElementQuadratureType = specfem::quadrature::lagrange_derivative<
      ngll, Tags::dimension_tag,
      Kokkos::DefaultExecutionSpace::scratch_memory_space,
      Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

  using PointTags = specfem::tags::Tags<Tags::dimension_tag, Tags::medium_tag,
                                        Tags::property_tag,
                                        Tags::attenuation_tag, using_simd>;

  using PointBoundaryType =
      specfem::point::boundary<Tags::boundary_tag, Tags::dimension_tag,
                               using_simd>;
  using PointDisplacementType = specfem::point::displacement<PointTags>;
  using PointVelocityType = specfem::point::velocity<PointTags>;
  using PointAccelerationType = specfem::point::acceleration<PointTags>;
  using PointJacobianMatrixType =
      specfem::point::jacobian_matrix<Tags::dimension_tag, true, using_simd>;
  using PointPropertyType = specfem::point::properties<PointTags>;
  using PointFieldDerivativesType =
      specfem::point::field_derivatives<PointTags>;

  using GradientPackType = specfem::point::stiffness_gradient_pack<
      attenuation_tag, typename PointFieldDerivativesType::value_type>;

  using PointWeightsType = specfem::point::weights<Tags::dimension_tag>;

  constexpr static int ncomp =
      specfem::element::attributes<dimension_tag, medium_tag>::components;

  constexpr static bool has_cosserat_couple_stress =
      specfem::element::attributes<dimension_tag,
                                   medium_tag>::has_cosserat_couple_stress;

  using AssemblyType = specfem::assembly::assembly<Tags::dimension_tag>;

  scatter_kernel(const AssemblyType &assembly, int istep)
      : assembly_(assembly), istep_(istep) {}

  static int shmem_size() {
    return ChunkFieldPackType::shmem_size() +
           ChunkAccelerationType::shmem_size() +
           ElementQuadratureType::shmem_size();
  }

  template <typename ChunkType> void operator()(ChunkType &chunk) const {
    const auto &mesh = assembly_.mesh;
    const auto &jacobian_matrix = assembly_.jacobian_matrix;
    const auto &properties = assembly_.properties;
    const auto &boundaries = assembly_.boundaries;
    const auto &attenuation = assembly_.attenuation;
    const auto &field =
        assembly_.fields.template get_simulation_field<Tags::wavefield_tag>();
    const auto &boundary_values =
        assembly_.boundary_values.template get_container<boundary_tag>();
    const int istep = istep_;

    specfem::execution::for_each_level(
        "specfem::compute::compute_stiffness_interaction::scatter",
        chunk.set_scratch_size(0, Kokkos::PerTeam(shmem_size())),
        KOKKOS_LAMBDA(
            const typename ChunkType::index_type &chunk_iterator_index) {
          const auto &chunk_index = chunk_iterator_index.get_index();
          const auto team = chunk_index.get_policy_index();

          ChunkFieldPackType field_pack(team.team_scratch(0));
          ElementQuadratureType lagrange_derivative(team);
          ChunkAccelerationType scratch_acc(team.team_scratch(0));

          // Kokkos scratch is NOT zero-initialized — explicit init pass
          specfem::execution::for_each_level(
              chunk_index.get_iterator(), [&](const auto &iterator_index) {
                const auto &local_index = iterator_index.get_local_index();
                for (int icomp = 0; icomp < ncomp; ++icomp)
                  scratch_acc(local_index, icomp) = static_cast<type_real>(0);
              });

          specfem::assembly::load_on_device(team, mesh, lagrange_derivative);
          specfem::assembly::load_on_device(chunk_index, field, field_pack);

          team.team_barrier();

          specfem::algorithms::gradient(
              chunk_index, jacobian_matrix, lagrange_derivative, field_pack,
              [&](const auto &iterator_index,
                  const GradientPackType &grad_pack) {
                const auto &index = iterator_index.get_index();
                const auto &local_index = iterator_index.get_local_index();

                PointJacobianMatrixType point_jacobian_matrix;
                specfem::assembly::load_on_device(index, jacobian_matrix,
                                                  point_jacobian_matrix);

                PointPropertyType point_property;
                specfem::assembly::load_on_device(index, properties,
                                                  point_property);

                PointWeightsType point_weights;
                specfem::assembly::load_on_device(index, mesh.weights,
                                                  point_weights);

                const auto field_derivatives =
                    grad_pack.template get_du<PointTags>();

                const auto field_derivatives_velocity =
                    grad_pack.template get_dv<PointTags>();

                PointDisplacementType point_displacement;
                specfem::assembly::load_on_device(index, field,
                                                  point_displacement);

                auto point_stress =
                    specfem::medium_physics::compute_stress<PointTags>(
                        point_property, field_derivatives);

                specfem::medium_physics::compute_attenuation<PointTags>(
                    index, point_stress, grad_pack, attenuation);

                specfem::medium_physics::compute_cosserat_stress(
                    point_property, point_displacement, point_stress);

                const auto F = point_stress * point_jacobian_matrix;

                specfem::algorithms::impl::scattered_divergence(
                    F, local_index, point_weights.product(),
                    lagrange_derivative, scratch_acc);

                if constexpr (has_cosserat_couple_stress) {
                  PointAccelerationType cosserat_delta;
                  for (int icomp = 0; icomp < ncomp; ++icomp)
                    cosserat_delta(icomp) = static_cast<type_real>(0);

                  specfem::medium_physics::compute_cosserat_couple_stress(
                      point_property,
                      point_weights.product() * point_jacobian_matrix.jacobian,
                      point_stress, cosserat_delta);

                  for (int icomp = 0; icomp < ncomp; ++icomp)
                    Kokkos::atomic_add(&scratch_acc(local_index, icomp),
                                       cosserat_delta(icomp));
                }
              });

          team.team_barrier();

          specfem::execution::for_each_level(
              chunk_index.get_iterator(), [&](const auto &iterator_index) {
                const auto &index = iterator_index.get_index();
                const auto &local_index = iterator_index.get_local_index();

                PointAccelerationType acceleration;
                for (int icomp = 0; icomp < ncomp; ++icomp)
                  acceleration(icomp) = -scratch_acc(local_index, icomp);

                PointPropertyType point_property;
                specfem::assembly::load_on_device(index, properties,
                                                  point_property);

                PointVelocityType velocity;
                specfem::assembly::load_on_device(index, field, velocity);

                PointBoundaryType point_boundary;
                specfem::assembly::load_on_device(index, boundaries,
                                                  point_boundary);

                PointWeightsType point_weights;
                specfem::assembly::load_on_device(index, mesh.weights,
                                                  point_weights);

                specfem::point::jacobian_matrix<dimension_tag, true, using_simd>
                    point_jacobian_matrix;
                specfem::assembly::load_on_device(index, jacobian_matrix,
                                                  point_jacobian_matrix);

                const auto factor =
                    point_weights.product() * point_jacobian_matrix.jacobian;

                specfem::medium_physics::compute_damping_force<decltype(factor),
                                                               PointTags>(
                    factor, point_property, velocity, acceleration);

                specfem::boundary_conditions::apply_boundary_conditions(
                    point_boundary, point_property, velocity, acceleration);

                if (Tags::wavefield_tag ==
                    specfem::simulation::field_type::forward) {
                  specfem::assembly::store_on_device(istep, index, acceleration,
                                                     boundary_values);
                }

                specfem::assembly::atomic_add_on_device(index, field,
                                                        acceleration);
              });
        });
  }

private:
  const AssemblyType &assembly_;
  int istep_;
};

} // namespace specfem::compute::impl
