#ifndef _SOURCE_HPP
#define _SOURCE_HPP

#include "compute/compute_mesh.hpp"
#include "compute/compute_partial_derivatives.hpp"
#include "compute/element_types/element_types.hpp"
#include "constants.hpp"
#include "enumerations/wavefield.hpp"
#include "kokkos_abstractions.h"
#include "quadrature/interface.hpp"
#include "source_time_function/interface.hpp"
#include "specfem_mpi/interface.hpp"
#include "specfem_setup.hpp"
#include "utilities/interface.hpp"
#include "yaml-cpp/yaml.h"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace sources {

/**
 * @brief Base source class
 *
 */
class source {

public:
  /**
   * @brief Default source constructor
   *
   */
  source(){};

  source(YAML::Node &Node, const int nsteps, const type_real dt);
  /**
   * @brief Get the x coordinate of the source
   *
   * @return type_real x-coordinate
   */
  type_real get_x() const { return x; }
  /**
   * @brief Get the z coordinate of the source
   *
   * @return type_real z-coordinate
   */
  type_real get_z() const { return z; }
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

  virtual void compute_source_array(
      const specfem::compute::mesh &mesh,
      const specfem::compute::partial_derivatives &partial_derivatives,
      const specfem::compute::element_types &element_types,
      specfem::kokkos::HostView3d<type_real> source_array) = 0;

  void compute_source_time_function(
      const type_real t0, const type_real dt, const int nsteps,
      specfem::kokkos::HostView2d<type_real> source_time_function) const {
    return this->forcing_function->compute_source_time_function(
        t0, dt, nsteps, source_time_function);
  }

  virtual specfem::wavefield::simulation_field get_wavefield_type() const = 0;

protected:
  type_real x; ///< x-coordinate of source
  type_real z; ///< z-coordinate of source

  std::unique_ptr<specfem::forcing_function::stf>
      forcing_function; ///< pointer to source time function
};

} // namespace sources

} // namespace specfem
#endif
