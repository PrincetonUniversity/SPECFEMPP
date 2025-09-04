#pragma once

#include "constants.hpp"
#include "enumerations/interface.hpp"
#include "kokkos_abstractions.h"
#include "quadrature/interface.hpp"
#include "source_time_function/interface.hpp"
#include "specfem/point.hpp"
#include "specfem/source.hpp"
#include "specfem_setup.hpp"
#include "utilities/interface.hpp"
#include "yaml-cpp/yaml.h"
#include <Kokkos_Core.hpp>

namespace specfem::sources {

/**
 * @brief Base class for all source types
 *
 * It is a container for source time functions, coordinates, and the medium
 * that the source is located in. This class provides the fundamental interface
 * common to all sources in SPECFEM++ simulations.
 *
 * @tparam DimensionTag The dimension specification (dim2 or dim3)
 *
 * @par Sources that inherit from this class:
 * - @ref specfem::sources::vector_source
 * - @ref specfem::sources::tensor_source
 *
 * @par Common Usage Pattern
 * @code
 * // All sources require a source time function
 * auto stf = std::make_unique<specfem::forcing_function::Ricker>(
 *     10.0,  // dominant frequency (Hz)
 *     0.01,  // time factor
 *     1.0,   // amplitude
 *     0.0,   // time shift
 *     1.0,   // normalization factor
 *     false  // do not reverse
 * );
 *
 * // Create any source (example with 2D force source)
 * auto source = specfem::sources::force<specfem::dimension::type::dim2>(
 *     5.0, 10.0,  // coordinates (x, z)
 *     0.0,        // angle
 *     std::move(stf),
 *     specfem::wavefield::simulation_field::forward
 * );
 *
 * // Common operations available for all sources:
 * source.set_medium_tag(specfem::element::medium_tag::elastic_psv);
 *
 * // Access coordinates
 * auto coords = source.get_global_coordinates();
 *
 * // Access timing information
 * type_real t0 = source.get_t0();
 * type_real tshift = source.get_tshift();
 *
 * // Update timing
 * source.update_tshift(1.5);
 *
 * // Check medium compatibility
 * auto supported_media = source.get_supported_media();
 * @endcode
 */
template <specfem::dimension::type DimensionTag> class source {

public:
  static constexpr auto dimension_tag = DimensionTag;
  /**
   * @brief Default source constructor
   *
   */
  source() {};

  /** @name 2D Constructors
   * @{
   */

  /**
   * @brief Construct a new 2D source object using the forcing function
   *
   * @param x x-coordinate of source
   * @param z z-coordinate of source
   * @param forcing_function pointer to source time function
   */
  template <specfem::dimension::type U = DimensionTag,
            typename std::enable_if<U == specfem::dimension::type::dim2>::type
                * = nullptr>
  source(type_real x, type_real z,
         std::unique_ptr<specfem::forcing_function::stf> forcing_function)
      : global_coordinates(x, z),
        forcing_function(std::move(forcing_function)){};

  /**
   * @brief Construct a new 2D source object from a YAML node and time steps
   *
   * @param Node YAML node containing source configuration
   * @param nsteps number of time steps
   * @param dt time step size
   */
  template <specfem::dimension::type U = DimensionTag,
            typename std::enable_if<U == specfem::dimension::type::dim2>::type
                * = nullptr>
  source(YAML::Node &Node, const int nsteps, const type_real dt);

  /** @} */

  /** @name 3D Constructors
   * @{
   */

  /**
   * @brief Construct a new 3D source object from a YAML node and time steps
   *
   * @param Node YAML node containing source configuration
   * @param nsteps number of time steps
   * @param dt time step size
   */
  template <specfem::dimension::type U = DimensionTag,
            typename std::enable_if<U == specfem::dimension::type::dim3>::type
                * = nullptr>
  source(YAML::Node &Node, const int nsteps, const type_real dt);

  /**
   * @brief Construct a new 3D source object using the forcing function
   *
   * @param x x-coordinate of source
   * @param y y-coordinate of source
   * @param z z-coordinate of source
   * @param forcing_function pointer to source time function
   */
  template <specfem::dimension::type U = DimensionTag,
            typename std::enable_if<U == specfem::dimension::type::dim3>::type
                * = nullptr>
  source(type_real x, type_real y, type_real z,
         std::unique_ptr<specfem::forcing_function::stf> forcing_function)
      : global_coordinates(x, y, z),
        forcing_function(std::move(forcing_function)){};

  /** @} */

  /**
   * @brief Get the value of t0 from the specfem::stf::stf object
   *
   * @return value of t0
   */
  type_real get_t0() const { return forcing_function->get_t0(); }

  type_real get_tshift() const { return forcing_function->get_tshift(); }
  /**
   * @brief Update the value of tshift for specfem::stf::stf object
   *
   * @return new value of tshift
   */
  void update_tshift(type_real tshift) {
    forcing_function->update_tshift(tshift);
  };
  /**
   * @brief User output
   *
   */
  virtual std::string print() const { return ""; };

  virtual ~source() = default;

  virtual source_type get_source_type() const = 0;

  void compute_source_time_function(
      const type_real t0, const type_real dt, const int nsteps,
      specfem::kokkos::HostView2d<type_real> source_time_function) const {
    return this->forcing_function->compute_source_time_function(
        t0, dt, nsteps, source_time_function);
  }

  virtual specfem::wavefield::simulation_field get_wavefield_type() const = 0;

  virtual bool operator==(const source &other) const {
    // Base implementation might just check type identity
    return typeid(*this) == typeid(other);
  }
  virtual bool operator!=(const source &other) const {
    return !(*this == other);
  }

  void set_forcing_function(YAML::Node &Node, const int nsteps,
                            const type_real dt);

  /**
   * @brief Get the forcing function object
   *
   * @return std::unique_ptr<specfem::forcing_function::stf>&
   */
  std::unique_ptr<specfem::forcing_function::stf> &get_forcing_function() {
    return forcing_function;
  }

  /**
   * @brief Set the local xi coordinates of the source in the local coordinate
   * system
   * @param specfem::point::local_coordinates<dimension_tag>
   * local_coordinates
   */
  void
  set_local_coordinates(const specfem::point::local_coordinates<dimension_tag>
                            &local_coordinates) {
    this->local_coordinates = local_coordinates;
  };

  /**
   * @brief Get the local coordinates of the source in the local coordinate
   * system
   * @return specfem::point::local_coordinates<dimension_tag>
   */
  specfem::point::local_coordinates<dimension_tag>
  get_local_coordinates() const {
    return local_coordinates;
  }

  /**
   * @brief Set the global coordinates of the source in the global coordinate
   * system
   * @param specfem::point::global_coordinates<dimension_tag> global_coordinates
   */
  void
  set_global_coordinates(const specfem::point::global_coordinates<dimension_tag>
                             &global_coordinates) {
    this->global_coordinates = global_coordinates;
  };

  /**
   * @brief Get the global coordinates of the source in the global coordinate
   * system
   * @return specfem::point::global_coordinates<dimension_tag>
   */
  specfem::point::global_coordinates<dimension_tag>
  get_global_coordinates() const {
    return global_coordinates;
  }

  /**
   * @brief Set the medium tag for the source.
   *
   * This needs to be set inside the since each medium requires a separate
   * implementation for each medium and some source do not have implementations
   * for certain media at all. E.g., if you want to assign a moment tensor to an
   * element in the water column (acoustic), it does not make sense, or rather
   * it is unphysical.
   *
   * @param medium_tag medium tag
   */
  void set_medium_tag(specfem::element::medium_tag medium_tag);

  /**
   * @brief Get the list of supported media for this source type
   *
   * @return std::vector<specfem::element::medium_tag> list of supported media
   */
  virtual std::vector<specfem::element::medium_tag>
  get_supported_media() const = 0;

  /**
   * @brief Get the medium tag for the source
   *
   * @return specfem::medium::medium_tag medium tag
   */
  specfem::element::medium_tag get_medium_tag() const { return medium_tag; }

protected:
  // Read-only member variables
  static constexpr const char *name =
      "!!! base_source, if this was printed, you are not using the "
      "correct source class !!!";

  std::unique_ptr<specfem::forcing_function::stf>
      forcing_function; ///< pointer to source time function

  // Member variables to be set.
  specfem::point::local_coordinates<dimension_tag>
      local_coordinates; ///< Local coordinates of the source in the local
  specfem::point::global_coordinates<dimension_tag>
      global_coordinates; ///< Global coordinates of the source in the global
                          ///< coordinate system
  specfem::element::medium_tag medium_tag;
};

} // namespace specfem::sources
