#pragma once

#include "specfem/algorithms.hpp"
#include "specfem/assembly/assembly.hpp"
#include "specfem/chunk_element.hpp"
#include "specfem/element.hpp"
#include "specfem/enums.hpp"
#include "specfem/execution.hpp"
#include "specfem/linear_system/element_stiffness.hpp"
#include "specfem/medium_physics.hpp"
#include "specfem/parallel_configuration.hpp"
#include "specfem/point.hpp"
#include "specfem/quadrature.hpp"
#include "specfem/tags.hpp"
#include <Kokkos_Core.hpp>

namespace specfem::linear_system_impl {

/**
 * @brief Kernel that fills dense element stiffness blocks by probing the
 * single-element operator with local unit vectors.
 *
 * For each column dof `jdof`, the kernel sets a unit displacement at that
 * dof (zero elsewhere), applies the matrix-free element operator
 * (gradient -> stress -> divergence), and stores the resulting integrated
 * internal-force values as column `jdof` of \f$ K_e \f$. Structure mirrors
 * `specfem::compute::impl::gather_kernel`.
 *
 * @tparam NGLL Number of GLL points per element edge
 * @tparam Tags Compile-time tags (dimension, medium, property, attenuation);
 *              attenuation must be `none` (pure stiffness, velocity == 0)
 */
template <int NGLL, typename Tags> class stiffness_probe_kernel {
public:
  constexpr static auto dimension_tag = Tags::dimension_tag;
  constexpr static auto medium_tag = Tags::medium_tag;
  constexpr static auto property_tag = Tags::property_tag;
  constexpr static auto attenuation_tag = Tags::attenuation_tag;
  constexpr static int ngll = NGLL;

  static_assert(dimension_tag == specfem::element::dimension_tag::dim3,
                "The stiffness probe kernel is implemented for 3D only.");
  static_assert(attenuation_tag == specfem::element::attenuation_tag::none,
                "The stiffness probe assumes a pure stiffness operator; "
                "attenuation would add a velocity-dependent term.");

  // Scalar on all backends: with SIMD, scratch and the divergence result
  // become simd packs (one lane per element), and scattering per-lane values
  // into k_e would need explicit lane extraction. Not worth it for a
  // setup-time probe.
  constexpr static bool using_simd = false;

  using simd = specfem::datatype::simd<type_real, using_simd>;

  using ParallelConfig = specfem::parallel_configuration::default_chunk_config<
      dimension_tag, simd, Kokkos::DefaultExecutionSpace>;

  using ChunkDisplacementType = specfem::chunk_element::displacement<
      ParallelConfig::chunk_size, ngll, dimension_tag, medium_tag, using_simd>;

  using ChunkStressIntegrandType = specfem::chunk_element::stress_integrand<
      ParallelConfig::chunk_size, ngll, dimension_tag, medium_tag,
      Kokkos::DefaultExecutionSpace::scratch_memory_space,
      Kokkos::MemoryTraits<Kokkos::Unmanaged>, using_simd>;

  using ElementQuadratureType = specfem::quadrature::lagrange_derivative<
      ngll, dimension_tag, Kokkos::DefaultExecutionSpace::scratch_memory_space,
      Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

  using PointTags = specfem::tags::Tags<dimension_tag, medium_tag, property_tag,
                                        attenuation_tag, using_simd>;

  using PointJacobianMatrixType =
      specfem::point::jacobian_matrix<dimension_tag, true, using_simd>;
  using PointPropertyType = specfem::point::properties<PointTags>;
  using PointFieldDerivativesType =
      specfem::point::field_derivatives<PointTags>;

  constexpr static int ncomp =
      specfem::element::attributes<dimension_tag, medium_tag>::components;

  /// Dense block edge length: rows/columns of \f$ K_e \f$
  constexpr static int ndof = ncomp * ngll * ngll * ngll;

  using AssemblyType = specfem::assembly::assembly<dimension_tag>;

  /**
   * @brief Construct the probe kernel.
   *
   * @param assembly Assembled mesh, jacobian matrix, and material properties
   * @param batch_begin_ispec First element index of the batch; element
   *        `ispec` writes its block to slot `ispec - batch_begin_ispec`
   * @param k_e Device buffer of shape `(>= batch size, ndof, ndof)`
   */
  using StiffnessViewType = Kokkos::View<type_real ***, Kokkos::LayoutRight,
                                         Kokkos::DefaultExecutionSpace>;

  stiffness_probe_kernel(const AssemblyType &assembly,
                         const int batch_begin_ispec,
                         const StiffnessViewType &k_e)
      : assembly_(assembly), batch_begin_ispec_(batch_begin_ispec), k_e_(k_e) {}

  /// Team scratch bytes required per chunk
  static int shmem_size() {
    return ChunkDisplacementType::shmem_size() +
           ChunkStressIntegrandType::shmem_size() +
           ElementQuadratureType::shmem_size();
  }

  template <typename ChunkType> void operator()(ChunkType &chunk) const {
    const auto &mesh = assembly_.mesh;
    const auto &jacobian_matrix = assembly_.jacobian_matrix;
    const auto &properties = assembly_.properties;
    const auto k_e = k_e_;
    const int batch_begin_ispec = batch_begin_ispec_;

    specfem::execution::for_each_level(
        "specfem::linear_system::stiffness_probe", chunk,
        KOKKOS_LAMBDA(
            const typename ChunkType::index_type &chunk_iterator_index) {
          const auto &chunk_index = chunk_iterator_index.get_index();
          const auto team = chunk_index.get_policy_index();

          ChunkDisplacementType displacement(team.team_scratch(0));
          ElementQuadratureType lagrange_derivative(team);
          ChunkStressIntegrandType stress_integrand(team);

          specfem::assembly::load_on_device(team, mesh, lagrange_derivative);

          team.team_barrier();

          for (int jdof = 0; jdof < ndof; ++jdof) {
            // Set the probe: zero displacement except a unit entry at the
            // local dof jdof of every element in the chunk.
            specfem::execution::for_each_level(
                chunk_index.get_iterator(), [&](const auto &iterator_index) {
                  const auto &local_index = iterator_index.get_local_index();
                  for (int icomp = 0; icomp < ncomp; ++icomp) {
                    displacement(local_index.ispec, local_index.iz,
                                 local_index.iy, local_index.ix, icomp) =
                        (specfem::linear_system::local_dof_index<ngll>(
                             icomp, local_index.iz, local_index.iy,
                             local_index.ix) == jdof)
                            ? static_cast<type_real>(1.0)
                            : static_cast<type_real>(0.0);
                  }
                });

            team.team_barrier();

            specfem::algorithms::gradient(
                chunk_index, jacobian_matrix, lagrange_derivative, displacement,
                [&](const auto &iterator_index, const auto &du) {
                  const auto &index = iterator_index.get_index();
                  const auto &local_index = iterator_index.get_local_index();

                  PointJacobianMatrixType point_jacobian_matrix;
                  specfem::assembly::load_on_device(index, jacobian_matrix,
                                                    point_jacobian_matrix);

                  PointPropertyType point_property;
                  specfem::assembly::load_on_device(index, properties,
                                                    point_property);

                  const PointFieldDerivativesType field_derivatives(du);

                  const auto point_stress =
                      specfem::medium_physics::compute_stress<PointTags>(
                          point_property, field_derivatives);

                  stress_integrand.F(local_index) =
                      point_stress * point_jacobian_matrix;
                });

            team.team_barrier();

            specfem::algorithms::divergence(
                chunk_index, mesh.weights, lagrange_derivative,
                stress_integrand.F,
                [&](const auto &iterator_index, const auto &result) {
                  const auto &index = iterator_index.get_index();
                  const auto &local_index = iterator_index.get_local_index();

                  const int ielement = index.ispec - batch_begin_ispec;
                  for (int icomp = 0; icomp < ncomp; ++icomp) {
                    k_e(ielement,
                        specfem::linear_system::local_dof_index<ngll>(
                            icomp, local_index.iz, local_index.iy,
                            local_index.ix),
                        jdof) = result(icomp);
                  }
                });

            // Protects the stress integrand: the next probe's gradient pass
            // writes F while this probe's divergence pass reads it.
            team.team_barrier();
          }
        });
  }

private:
  const AssemblyType &assembly_;
  int batch_begin_ispec_;
  StiffnessViewType k_e_;
};

} // namespace specfem::linear_system_impl
