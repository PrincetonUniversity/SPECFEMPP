#pragma once

#include "enumerations/interface.hpp"
#include "specfem/data_access.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace point {

/**
 * @brief Source term representation for external forcing in spectral element
 * simulations.
 *
 * The source class encapsulates external force terms applied at quadrature
 * points within spectral elements, representing physical phenomena such as
 * seismic sources, acoustic sources, or external loads. These source terms are
 * essential components of the governing equations in wave propagation problems
 * and contribute to the right-hand side of the semi-discrete system of
 * equations.
 *
 * @tparam DimensionTag Spatial dimension of the source term:
 *                      - `dim2`: 2D sources with x,z components
 *                      - `dim3`: 3D sources with x,y,z components
 * @tparam MediumTag Physical medium determining source interpretation:
 *                  - `acoustic`: Scalar pressure or potential sources
 *                  - `elastic`: Vector displacement or force sources
 *                  - `poroelastic`: Coupled solid-fluid sources
 * @tparam WavefieldType Target wavefield for source application:
 *                       - `forward`: Forward wavefield simulation
 *                       - `adjoint`: Adjoint/backward wavefield
 *                       - `kernel`: Sensitivity kernel computation
 *
 * @note Source terms are typically time-dependent and require careful temporal
 *       discretization to maintain accuracy and stability.
 *
 * @see specfem::sources for source initialization and management
 * @see specfem::time_marching for temporal integration with sources
 *
 * @code
 * // Example: Applying a 2D elastic point force source
 * using ElasticSource = specfem::point::source<
 *     specfem::dimension::type::dim2,
 *     specfem::element::medium_tag::elastic,
 *     specfem::wavefield::simulation_field::forward>;
 *
 * ElasticSource point_force;
 *
 * // Set force components (Newtons)
 * point_force.force_vector(0) = 1000.0;  // x-direction force
 * point_force.force_vector(1) = 0.0;     // z-direction force
 *
 * // Apply source term in assembly
 * type_real source_contribution = point_force.force_vector(icomp) *
 *                                 basis_function * time_function;
 * rhs_vector(iglob) += source_contribution * quadrature_weight;
 *
 * // Example: Acoustic pressure source
 * using AcousticSource = specfem::point::source<
 *     specfem::dimension::type::dim2,
 *     specfem::element::medium_tag::acoustic,
 *     specfem::wavefield::simulation_field::forward>;
 *
 * AcousticSource pressure_source;
 * pressure_source.pressure_amplitude = ricker_wavelet(time);
 * @endcode
 */
template <specfem::dimension::type DimensionTag,
          specfem::element::medium_tag MediumTag,
          specfem::wavefield::simulation_field WavefieldType>
struct source
    : public specfem::data_access::Accessor<
          specfem::data_access::AccessorType::point,
          specfem::data_access::DataClassType::source, DimensionTag, false> {
private:
  using base_type = specfem::data_access::Accessor<
      specfem::data_access::AccessorType::point,
      specfem::data_access::DataClassType::source, DimensionTag,
      false>; ///< Base type for the
              ///< source
public:
  constexpr static auto medium_tag = MediumTag; ///< Medium tag of the spectral
                                                ///< element
  constexpr static auto wavefield_tag = WavefieldType; ///< Wavefield type on
                                                       ///< which the source is
                                                       ///< applied

  constexpr static int components =
      specfem::element::attributes<DimensionTag,
                                   MediumTag>::components; ///< Number
                                                           ///< of
                                                           ///< components
                                                           ///< in
                                                           ///< the
                                                           ///< medium

  using value_type =
      typename base_type::template vector_type<type_real,
                                               components>; ///<
                                                            ///< Value
                                                            ///< type
                                                            ///< to
                                                            ///< store
                                                            ///< source
                                                            ///< information

  value_type stf;                  ///< Source time function
  value_type lagrange_interpolant; ///< Lagrange interpolant

  KOKKOS_INLINE_FUNCTION source() = default;

  /**
   * @brief Constructor
   *
   * @param stf Source time function
   * @param lagrange_interpolant Lagrange interpolant
   *
   */
  KOKKOS_INLINE_FUNCTION source(const value_type &stf,
                                const value_type &lagrange_interpolant)
      : stf(stf), lagrange_interpolant(lagrange_interpolant) {}
};

} // namespace point
} // namespace specfem
