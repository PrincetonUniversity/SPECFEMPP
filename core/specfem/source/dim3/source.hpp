#pragma once

#include "constants.hpp"
#include "enumerations/interface.hpp"
#include "kokkos_abstractions.h"
#include "source_time_function/interface.hpp"
#include "specfem/source.hpp"
#include "specfem_setup.hpp"
#include "utilities/interface.hpp"
#include "yaml-cpp/yaml.h"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace sources {

template <> class source<specfem::dimension::type::dim3> {

public:
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
      : x(x), y(y), z(z), forcing_function(std::move(forcing_function)) {};

  /**
   * @brief Construct a new source object from a YAML node and time steps
   *
   * @param Node
   * @param nsteps
   * @param dt
   */
  source(YAML::Node &Node, const int nsteps, const type_real dt);

  /**
   * @brief Get the x coordinate of the source
   *
   * @return type_real x-coordinate
   */
  type_real get_x() const { return x; }

  /**
   * @brief Get the y coordinate of the source
   *
   * @return type_real y-coordinate
   */
  type_real get_y() const { return y; }

  /**
   * @brief Get the z coordinate of the source
   *
   * @return type_real z-coordinate
   */
  type_real get_z() const { return z; }

  /**
   * @brief Get coordinates as a Kokkos array
   *
   * @return Kokkos::View<type_real[2], Kokkos::HostSpace> coordinates array [x,
   * z]
   */
  Kokkos::View<type_real[3], Kokkos::HostSpace> get_coords() const {
    Kokkos::View<type_real[3], Kokkos::HostSpace> coords("coords");
    coords[0] = x;
    coords[1] = y;
    coords[2] = z;
    return coords;
  }

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
   * @param xi xi coordinate of source in the local coordinate system
   */
  void set_xi(type_real xi) { this->xi = xi; }

  /**
   * @brief Set the local eta coordinates of the source in the local
   * coordinate system
   * @param eta eta coordinate of source in the local coordinate system
   */
  void set_eta(type_real eta) { this->eta = eta; }

  /**
   * @brief Set the local zeta coordinates of the source in the local
   * coordinate system
   * @param zeta zeta coordinate of source in the local coordinate system
   */
  void set_zeta(type_real zeta) { this->zeta = zeta; }

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
   * @brief Get the local xi coordinate of the source in the local coordinate
   * system
   *
   * @return type_real xi coordinate
   */
  type_real get_xi() const { return this->xi; }
  /**
   * @brief Get the local eta coordinate of the source in the local coordinate
   * system
   *
   * @return type_real eta coordinate
   */
  type_real get_eta() const { return this->eta; }

  /**
   * @brief Get the local zeta coordinate of the source in the local coordinate
   * system
   *
   * @return type_real zeta coordinate
   */
  type_real get_zeta() const { return this->zeta; }

  /**
   * @brief Get the medium tag for the source
   *
   * @return specfem::medium::medium_tag medium tag
   */
  specfem::element::medium_tag get_medium_tag() const { return medium_tag; }
  /**
   * @brief Get the index of the element that the source is located in
   *
   * @return int index of the element
   */
  int get_element_index() const { return this->element_index; }
  /**
   * @brief Set the index of the element that the source is located in
   *
   * @param ielement index of the element
   */
  void set_element_index(int element_index) {
    this->element_index = element_index;
  }

protected:
  // Read-only member variables
  std::string name = "base_source, if this was printed, you are not using the "
                     "correct source class";
  type_real x; ///< x-coordinate of source
  type_real y; ///< y-coordinate of source
  type_real z; ///< z-coordinate of source
  std::unique_ptr<specfem::forcing_function::stf>
      forcing_function; ///< pointer to source time function

  // Member variables to be set.

  type_real xi;
  ///< xi coordinate of source in the local coordinate system

  type_real eta;
  ///< eta coordinate of source in the local coordinate system

  type_real zeta;
  ///< zeta coordinate of source in the local coordinate system

  specfem::element::medium_tag medium_tag;
  ///< medium tag for the source
  ///< (e.g., acoustic, elastic, poroelastic, etc.)

  int element_index; ///< index of the element that the source is located in
};

} // namespace sources

} // namespace specfem
