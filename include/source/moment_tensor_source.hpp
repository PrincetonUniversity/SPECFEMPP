#ifndef _MOMENT_TENSOR_SOURCE_HPP
#define _MOMENT_TENSOR_SOURCE_HPP

#include "compute/compute_mesh.hpp"
#include "compute/compute_partial_derivatives.hpp"
#include "compute/element_types/element_types.hpp"
#include "constants.hpp"
#include "enumerations/specfem_enums.hpp"
#include "kokkos_abstractions.h"
#include "quadrature/interface.hpp"
#include "source.hpp"
#include "source_time_function/interface.hpp"
#include "specfem_mpi/interface.hpp"
#include "specfem_setup.hpp"
#include "utilities/interface.hpp"
#include "yaml-cpp/yaml.h"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace sources {
/**
 * @brief Moment-tensor source
 *
 */
class moment_tensor : public source {

public:
  /**
   * @brief Default source constructor
   *
   */
  moment_tensor(){};

  /**
   * @brief Get the Mxx component of the moment tensor
   *
   * @return type_real x-coordinate
   */
  type_real get_Mxx() const { return Mxx; }
  /**
   * @brief Get the Mxz component of the moment tensor
   *
   * @return type_real z-coordinate
   */
  type_real get_Mxz() const { return Mxz; }
  /**
   * @brief Get the Mzz component of the moment tensor
   *
   * @return type_real z-coordinate
   */
  type_real get_Mzz() const { return Mzz; }

  /**
   * @brief Construct a new moment tensor force object
   *
   * @param moment_tensor a moment_tensor data holder read from source file
   * written in .yml format
   */
  moment_tensor(YAML::Node &Node, const int nsteps, const type_real dt,
                const specfem::wavefield::simulation_field wavefield_type)
      : Mxx(Node["Mxx"].as<type_real>()), Mzz(Node["Mzz"].as<type_real>()),
        Mxz(Node["Mxz"].as<type_real>()),
        wavefield_type(wavefield_type), specfem::sources::source(Node, nsteps,
                                                                 dt){};
  /**
   * @brief User output
   *
   */
  std::string print() const override;

  void compute_source_array(
      const specfem::compute::mesh &mesh,
      const specfem::compute::partial_derivatives &partial_derivatives,
      const specfem::compute::element_types &element_types,
      specfem::kokkos::HostView3d<type_real> source_array) override;

  specfem::wavefield::simulation_field get_wavefield_type() const override {
    return wavefield_type;
  }

private:
  type_real Mxx;                                       ///< Mxx for the source
  type_real Mxz;                                       ///< Mxz for the source
  type_real Mzz;                                       ///< Mzz for the source
  specfem::wavefield::simulation_field wavefield_type; ///< Type of wavefield on
                                                       ///< which the source
                                                       ///< acts
};
} // namespace sources
} // namespace specfem

#endif
