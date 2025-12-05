#pragma once

#include "enumerations/interface.hpp"
#include "specfem/data_access.hpp"
#include "specfem_setup.hpp"
#include <Kokkos_Core.hpp>

namespace specfem::point {

/**
 * @brief Multi-physics interface coupling for spectral element domain
 * decomposition.
 *
 * The coupled_interface class manages data and computations at interfaces
 * between different physical media in multi-physics spectral element
 * simulations. These interfaces are critical for accurately modeling wave
 * propagation across material boundaries where different governing equations
 * apply, such as fluid-solid interfaces in seismic and acoustic applications.
 *
 * **Connection Types:**
 * - **Strongly conforming**: Matching mesh topology across interface
 * - **Weakly conforming**: Non-matching meshes requiring interpolation
 * - **Mortar methods**: Constraint enforcement through Lagrange multipliers
 *
 * @tparam DimensionTag Spatial dimension of the interface geometry:
 *                      - `dim2`: 2D interfaces (lines in 2D domains)
 *                      - `dim3`: 3D interfaces (surfaces in 3D domains)
 * @tparam ConnectionTag Mesh conformity specification:
 *                       - `strongly_conforming`: Perfect mesh alignment
 *                       - `weakly_conforming`: Non-matching discretizations
 * @tparam InterfaceTag Physical media coupling type:
 *                     - `elastic_acoustic`: Elastic to acoustic transition
 *                     - `acoustic_elastic`: Acoustic to elastic transition
 *                     - `poroelastic_elastic`: Porous to solid transition
 * @tparam BoundaryTag Boundary condition enforcement at interface:
 *                    - `none`: Internal interface (no external BC)
 *                    - `acoustic_free_surface`: Free surface conditions
 *                    - `stacey`: Absorbing boundary conditions
 *
 * @note Interface coupling requires careful treatment of field variable
 *       transformations and conservation laws across material boundaries.
 *
 * @see specfem::coupling for interface operator implementations
 * @see specfem::mortar for non-conforming interface methods
 *
 * @code
 * // Example: Elastic-acoustic interface setup
 * using FluidSolidInterface = specfem::point::coupled_interface<
 *     specfem::dimension::type::dim2,
 *     specfem::connections::type::strongly_conforming,
 *     specfem::interface::interface_tag::elastic_acoustic,
 *     specfem::element::boundary_tag::none>;
 *
 * FluidSolidInterface interface_pt;
 *
 * // Load interface geometry data
 * specfem::assembly::load_on_device(index, interface_data, interface_pt);
 *
 * // Apply interface coupling conditions
 * type_real normal_stress_fluid = fluid_pressure * interface_pt.edge_normal(0);
 * type_real normal_stress_solid =
 * solid_stress_tensor.contract(interface_pt.edge_normal);
 *
 * // Enforce stress continuity condition
 * assert(std::abs(normal_stress_fluid - normal_stress_solid) < tolerance);
 * @endcode
 */
template <specfem::dimension::type DimensionTag,
          specfem::connections::type ConnectionTag,
          specfem::interface::interface_tag InterfaceTag,
          specfem::element::boundary_tag BoundaryTag>
struct coupled_interface;

/**
 * @brief 2D specialization for multi-physics interface coupling points.
 *
 * This specialization handles coupling between different physical media along
 * 1D interfaces (edges) within 2D spectral element domains. It stores the
 * geometric and physical data necessary for enforcing interface conditions
 * between adjacent elements with different governing equations.
 *
 * **2D Interface Geometry:**
 * The interface is represented as an edge in 2D space with:
 * - Edge factor: Integration weight for interface integrals
 * - Normal vector: Outward unit normal to the interface edge
 * - Tangent vector: Derived from normal for tangential computations
 *
 * **Interface Conditions (2D):**
 * For elastic-acoustic coupling:
 * - Normal stress continuity: \f$T_{nn}^{(elastic)} = -p^{(acoustic)}\f$
 * - Normal velocity continuity: \f$v_n^{(elastic)} = v_n^{(acoustic)}\f$
 * - Zero tangential stress on fluid side: \f$T_{nt}^{(acoustic)} = 0\f$
 *
 * **Common 2D Applications:**
 * - Ocean-bottom seismology: Water-sediment interfaces in marine surveys
 * - Exploration geophysics: Fluid-filled fractures and cavities
 * - Atmospheric acoustics: Air-ground coupling for blast waves
 * - Medical imaging: Tissue-fluid boundaries in ultrasonic applications
 *
 * @tparam ConnectionTag Mesh conformity across the interface:
 *                       - Strongly conforming interfaces for exact coupling
 *                       - Weakly conforming for adaptive mesh refinement
 * @tparam InterfaceTag Physics coupling specification defining field
 * relationships and conservation laws across the material boundary.
 * @tparam BoundaryTag External boundary condition if interface is at domain
 * edge, otherwise specifies internal interface treatment.
 *
 * @note 2D interfaces require careful handling of in-plane vs out-of-plane
 *       wave components, especially for elastic domains with SH-wave coupling.
 *
 * @see specfem::interface::operators for 2D coupling operator implementations
 *
 * @code
 * // Example: 2D seabed interface (water-sediment)
 * using SeabedInterface = specfem::point::coupled_interface<
 *     specfem::dimension::type::dim2,
 *     specfem::connections::type::strongly_conforming,
 *     specfem::interface::interface_tag::acoustic_elastic,
 *     specfem::element::boundary_tag::none>;
 *
 * SeabedInterface seabed_pt;
 *
 * // Set interface normal pointing into water
 * seabed_pt.edge_normal(0) = 0.0;  // horizontal component
 * seabed_pt.edge_normal(1) = 1.0;  // vertical component (upward)
 * seabed_pt.edge_factor = edge_integration_weight;
 *
 * // Apply water pressure to sediment
 * type_real water_pressure = acoustic_field.pressure;
 * type_real normal_traction = -water_pressure * seabed_pt.edge_normal(1);
 *
 * // Enforce velocity continuity at interface
 * sediment_velocity.normal_component = water_velocity.normal_component;
 * @endcode
 */
template <specfem::connections::type ConnectionTag,
          specfem::interface::interface_tag InterfaceTag,
          specfem::element::boundary_tag BoundaryTag>
struct coupled_interface<specfem::dimension::type::dim2, ConnectionTag,
                         InterfaceTag, BoundaryTag>
    : public specfem::data_access::Accessor<
          specfem::data_access::AccessorType::point,
          specfem::data_access::DataClassType::coupled_interface,
          specfem::dimension::type::dim2, false> {
private:
  /** @brief Base accessor type alias */
  using base_type = specfem::data_access::Accessor<
      specfem::data_access::AccessorType::point,
      specfem::data_access::DataClassType::coupled_interface,
      specfem::dimension::type::dim2, false>;

public:
  /** @brief Dimension tag for 2D specialization */
  static constexpr auto dimension_tag = specfem::dimension::type::dim2;
  /** @brief Connection type between elements */
  static constexpr auto connection_tag = ConnectionTag;
  /** @brief Interface type (elastic-acoustic or acoustic-elastic) */
  static constexpr auto interface_tag = InterfaceTag;
  /** @brief Boundary condition type */
  static constexpr auto boundary_tag = BoundaryTag;

  /** @brief Edge scaling factor for interface computations */
  scalar_type<type_real> edge_factor;
  /** @brief Edge normal vector (2D) */
  vector_type<type_real, 2> edge_normal;

  /**
   * @brief Constructs coupled interface point with geometric data
   *
   * @param edge_factor Scaling factor for the interface edge
   * @param edge_normal_ Normal vector at the interface edge
   */
  KOKKOS_INLINE_FUNCTION
  coupled_interface(const scalar_type<type_real> &edge_factor,
                    const vector_type<type_real, 2> &edge_normal_)
      : edge_factor(edge_factor), edge_normal(edge_normal_) {}

  KOKKOS_INLINE_FUNCTION
  coupled_interface() = default;
};

} // namespace specfem::point
