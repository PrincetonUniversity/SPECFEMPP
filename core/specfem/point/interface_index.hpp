
#pragma once

#include "edge_index.hpp"
#include "enumerations/interface.hpp"

namespace specfem::point {

/**
 * @brief Index pair for coupled interface points in multi-physics spectral
 * element simulations.
 *
 * The interface_index class provides a sophisticated indexing mechanism for
 * managing the coupling between different physical media in multi-physics
 * problems. Each interface point requires knowledge of corresponding locations
 * on both sides of the interface to properly enforce continuity conditions and
 * exchange field information.
 *
 * In multi-physics problems, interfaces represent boundaries between different
 * governing equations (e.g., acoustic wave equation in fluids, elastic wave
 * equation in solids). Proper coupling requires:
 * 1. **Kinematic continuity**: Normal components of displacement/velocity must
 * be continuous
 * 2. **Dynamic continuity**: Normal stresses/pressures must be continuous
 * 3. **Geometric correspondence**: Points on both sides must be spatially
 * aligned
 *
 * The mathematical coupling conditions at an acoustic-elastic interface are:
 * \f$
 *   \mathbf{v}_f \cdot \mathbf{n} = \dot{\mathbf{u}}_s \cdot \mathbf{n}
 * \f$
 * \f$
 *   p_f = -\boldsymbol{\sigma}_s \mathbf{n} \cdot \mathbf{n}
 * \f$
 * where \f$\mathbf{v}_f\f$ is fluid velocity, \f$\dot{\mathbf{u}}_s\f$ is solid
 * velocity,
 * \f$p_f\f$ is fluid pressure, \f$\boldsymbol{\sigma}_s\f$ is solid stress
 * tensor, and \f$\mathbf{n}\f$ is the interface normal.
 *
 * @tparam DimensionTag Spatial dimension of the interface geometry.
 *                      - `specfem::dimension::type::dim2` for 2D interfaces
 * (lines)
 *                      - `specfem::dimension::type::dim3` for 3D interfaces
 * (surfaces)
 *
 * @note Interface indices are computed during the mesh preprocessing stage and
 *       remain constant throughout the simulation.
 *
 * @see specfem::coupling::acoustic_elastic_coupling
 * @see specfem::point::edge_index
 * @see specfem::enumerations::interface
 *
 * @code
 * // Example: Setting up acoustic-elastic interface coupling
 * using InterfaceIndex2D = specfem::point::interface_index<
 *     specfem::dimension::type::dim2>;
 *
 * // Create edge indices for both sides
 * auto acoustic_edge = specfem::point::edge_index<dim2>(elem_acoustic, face_id,
 * ngll_point); auto elastic_edge =
 * specfem::point::edge_index<dim2>(elem_elastic, face_id, ngll_point);
 *
 * // Couple the interface points
 * InterfaceIndex2D interface_coupling(acoustic_edge, elastic_edge);
 *
 * // Access coupled indices
 * auto fluid_point = interface_coupling.self_index;    // Acoustic side
 * auto solid_point = interface_coupling.coupled_index; // Elastic side
 *
 * // Use for field exchange:
 * // fluid_pressure = extract_pressure(fluid_point);
 * // solid_velocity = extract_velocity(solid_point);
 * // enforce_coupling_condition(fluid_pressure, solid_velocity, normal);
 * @endcode
 */
template <specfem::dimension::type DimensionTag> class interface_index {
public:
  /**
   * @brief Edge index on the self side of the coupled interface.
   *
   * Identifies the quadrature point location on the "primary" side of the
   * interface, typically corresponding to the element from which coupling is
   * being evaluated. This could be either the acoustic or elastic side
   * depending on the coupling direction and implementation strategy.
   *
   * @note The self/coupled distinction is context-dependent and determined by
   *       the specific coupling algorithm being used.
   */
  specfem::point::edge_index<DimensionTag> self_index;

  /**
   * @brief Edge index on the coupled side of the interface.
   *
   * Identifies the corresponding quadrature point on the "secondary" side of
   * the interface, representing the point that must be coupled with the
   * self_index point to enforce proper interface conditions. The correspondence
   * is established through geometric projection and interpolation during mesh
   * preprocessing.
   *
   * @note Coupled indices may require interpolation if the meshes on both sides
   *       have different resolution or are non-conforming.
   */
  specfem::point::edge_index<DimensionTag> coupled_index;

  /**
   * @brief Default constructor for uninitialized interface index.
   *
   * Creates an interface index with default-initialized edge indices on both
   * sides. The resulting object requires explicit assignment of valid edge
   * indices before use in coupling operations.
   *
   * @note Marked with `KOKKOS_INLINE_FUNCTION` for device/host portability.
   */
  KOKKOS_INLINE_FUNCTION
  interface_index() = default;

  /**
   * @brief Constructs interface index from corresponding edge indices on both
   * sides.
   *
   * Creates a fully initialized interface index that establishes the geometric
   * correspondence between quadrature points on different sides of a
   * multi-physics interface. This constructor is typically used during mesh
   * preprocessing when interface coupling relationships are established.
   *
   * @param self_index Edge index identifying the quadrature point on the
   * primary side of the interface (element, face, and local point).
   * @param coupled_index Edge index identifying the corresponding quadrature
   * point on the secondary side of the interface.
   *
   * @note The order of self_index and coupled_index may affect the sign
   * convention used in coupling condition enforcement, depending on the
   * specific physics implementation.
   *
   * @code
   * // Example: Coupling acoustic element 42, face 1, point 3 with
   * //          elastic element 87, face 2, point 5
   * auto acoustic_edge = edge_index<dim2>(42, 1, 3);
   * auto elastic_edge = edge_index<dim2>(87, 2, 5);
   * auto coupling = interface_index<dim2>(acoustic_edge, elastic_edge);
   * @endcode
   */
  KOKKOS_INLINE_FUNCTION
  interface_index(const specfem::point::edge_index<DimensionTag> &self_index,
                  const specfem::point::edge_index<DimensionTag> &coupled_index)
      : self_index(self_index), coupled_index(coupled_index) {}
};

} // namespace specfem::point
