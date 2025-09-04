#pragma once

#include "enumerations/interface.hpp"
#include "specfem/source.hpp"

namespace specfem {
namespace sources {

/**
 * @brief Class representing a vector source
 *
 * The vector source class is a base class for all vector sources in the
 * simulation. It provides the common interface and functionality for
 * manipulating vector sources. The main functionality being the return of a
 * vector that can be used to compute the GLL level source array, which is
 * applied in the simulation.
 *
 * Vector sources apply forces in specific directions and the dimensionality
 * of the force vector depends on the medium type and wave field configuration.
 *
 * @tparam DimensionTag The dimension specification (dim2 or dim3)
 *
 * @par Examples of vector sources:
 * - @ref specfem::sources::force - Directional force sources
 * - @ref specfem::sources::external - External boundary sources
 * - @ref specfem::sources::adjoint_source - Adjoint sources for inversion
 * - @ref specfem::sources::cosserat_force - Cosserat elastic sources
 *
 * @par Vector Source Usage Pattern
 * @code
 * // Example: Creating and using a vector source (2D force)
 * auto stf = std::make_unique<specfem::forcing_function::Ricker>(
 *     15.0, 0.01, 1.0, 0.0, 1.0, false
 * );
 *
 * auto vector_src = specfem::sources::force<specfem::dimension::type::dim2>(
 *     3.5, 7.2,   // coordinates (x, z)
 *     30.0,       // angle in degrees
 *     std::move(stf),
 *     specfem::wavefield::simulation_field::forward
 * );
 *
 * // Set the medium where the source is located
 * vector_src.set_medium_tag(specfem::element::medium_tag::elastic_psv);
 *
 * // Get the force vector - size depends on medium:
 * // - acoustic: 1 component (pressure)
 * // - elastic_sh: 1 component (out-of-plane)
 * // - elastic_psv: 2 components (in-plane x,z)
 * // - elastic_psv_t: 3 components (x,z + rotation)
 * auto force_vector = vector_src.get_force_vector();
 *
 * // All vector sources return vector_source type
 * assert(vector_src.get_source_type() ==
 *        specfem::sources::source_type::vector_source);
 * @endcode
 *
 * @note This class inherits from @ref specfem::sources::source
 */
template <specfem::dimension::type DimensionTag>
class vector_source : public source<DimensionTag> {

public:
  /**
   * @brief Default vector source constructor
   *
   */
  vector_source() {};

  /**
   * @brief Construct a new 2D vector source object using the forcing function
   *
   * @param x x-coordinate of source
   * @param z z-coordinate of source
   * @param forcing_function pointer to source time function
   * @param wavefield_type type of wavefield
   */
  template <specfem::dimension::type U = DimensionTag,
            typename std::enable_if<U == specfem::dimension::type::dim2>::type
                * = nullptr>
  vector_source(
      type_real x, type_real z,
      std::unique_ptr<specfem::forcing_function::stf> forcing_function)
      : source<DimensionTag>(x, z, std::move(forcing_function)){};

  /**
   * @brief Construct a new 3D vector source object using the forcing function
   * @param x x-coordinate of source
   * @param y y-coordinate of source
   * @param z z-coordinate of source
   * @param forcing_function pointer to source time function
   * @param wavefield_type type of wavefield
   */
  template <specfem::dimension::type U = DimensionTag,
            typename std::enable_if<U == specfem::dimension::type::dim3>::type
                * = nullptr>
  vector_source(
      type_real x, type_real y, type_real z,
      std::unique_ptr<specfem::forcing_function::stf> forcing_function)
      : source<DimensionTag>(x, y, z, std::move(forcing_function)){};

  /**
   * @brief Construct a new vector source object from a YAML node and time steps
   *
   * @param Node YAML node defining the vector source
   * @param nsteps Number of time steps in the simulation
   * @param dt Time increment in the simulation
   * @param wavefield_type Type of wavefield on which the source acts
   */
  vector_source(YAML::Node &Node, const int nsteps, const type_real dt)
      : source<DimensionTag>(Node, nsteps, dt) {};

  /**
   * @brief Get the source type object
   *
   * @return source_type
   */
  source_type get_source_type() const override {
    return source_type::vector_source;
  }

  /**
   * @brief Get the force vector
   *
   * Returns the force vector \f$\mathbf{f}\f$ applied by this vector source.
   * The dimensionality and components depend on the medium type and simulation
   * dimension.
   *
   * @par Mathematical Definition
   * The force vector represents the body force applied to the medium:
   * \f[
   * \mathbf{f} = \begin{cases}
   * [f_p] & \text{acoustic: pressure source} \\
   * [f_y] & \text{elastic SH (2D): out-of-plane force} \\
   * [f_x, f_z] & \text{elastic PSV (2D): in-plane forces} \\
   * [f_x, f_z, m_y] & \text{elastic PSV+T (2D): forces + rotational moment} \\
   * [f_x, f_y, f_z] & \text{elastic (3D): three force components} \\
   * [f_x, f_y, f_z, m_x, m_y, m_z] & \text{elastic Cosserat (3D): forces +
   * moments}
   * \end{cases}
   * \f]
   *
   * where:
   * - \f$f_i\f$ are force components in direction \f$i\f$
   * - \f$m_i\f$ are rotational moment components about axis \f$i\f$
   * - \f$f_p\f$ is the pressure source amplitude
   *
   * @note The actual force components and their physical meaning depend on the
   * specific source type implementation. See individual source classes for
   * detailed mathematical definitions of their force vectors.
   *
   * @return Kokkos::View<type_real *, Kokkos::LayoutLeft, Kokkos::HostSpace>
   * Force vector with size depending on medium type
   */
  virtual specfem::kokkos::HostView1d<type_real> get_force_vector() const = 0;
};

} // namespace sources
} // namespace specfem
