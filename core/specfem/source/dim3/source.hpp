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

namespace specfem {
namespace sources {

template <> class source<specfem::dimension::type::dim3> {

public:
  static constexpr auto dimension_tag = specfem::dimension::type::dim3;
  /**
   * @brief Default source constructor
   *
   */
  source() {};

  /**
   * @brief Construct a new source object using the forcing function
   *
   * @param x x-coordinate of source
   * @param y y-coordinate of source
   * @param z z-coordinate of source
   * @param forcing_function pointer to source time function
   * @param wavefield_type type of wavefield
   */
  source(type_real x, type_real y, type_real z,
         std::unique_ptr<specfem::forcing_function::stf> forcing_function)
      : global_coordinates(x, y, z),
        forcing_function(std::move(forcing_function)) {};

  /**
   * @brief Construct a new source object from a YAML node and time steps
   *
   * @param Node
   * @param nsteps
   * @param dt
   */
  source(YAML::Node &Node, const int nsteps, const type_real dt);

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
   * @param specfem::point::local_coordinates<specfem::dimension::type::dim3>
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
   * @return specfem::point::local_coordinates<specfem::dimension::type::dim3>
   */
  specfem::point::local_coordinates<dimension_tag>
  get_local_coordinates() const {
    return local_coordinates;
  }

  /**
   * @brief Set the global coordinates of the source in the global coordinate
   * system
   * @param specfem::point::global_coordinates<specfem::dimension::type::dim3>
   * global_coordinates
   */
  void
  set_global_coordinates(const specfem::point::global_coordinates<dimension_tag>
                             &global_coordinates) {
    this->global_coordinates = global_coordinates;
  };

  /**
   * @brief Get the global coordinates of the source in the global coordinate
   * system
   * @return specfem::point::global_coordinates<specfem::dimension::type::dim3>
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
      "3D base_source, if this was printed, you are not using the "
      "correct source class";
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

} // namespace sources

} // namespace specfem
