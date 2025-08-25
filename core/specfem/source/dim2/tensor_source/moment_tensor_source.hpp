#pragma once

#include "constants.hpp"
#include "enumerations/interface.hpp"
#include "kokkos_abstractions.h"
#include "quadrature/interface.hpp"
#include "source_time_function/interface.hpp"
#include "specfem/source.hpp"
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
template <>
class moment_tensor<specfem::dimension::type::dim2>
    : public tensor_source<specfem::dimension::type::dim2> {

public:
  /**
   * @brief Default source constructor
   *
   */
  moment_tensor() {};

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
        Mxz(Node["Mxz"].as<type_real>()), wavefield_type(wavefield_type),
        tensor_source<specfem::dimension::type::dim2>(Node, nsteps, dt) {};

  /**
   * @brief Costruct new moment tensor source using forcing function
   *
   * @param x x-coordinate of source
   * @param z z-coordinate of source
   * @param Mxx Mxx component of moment tensor
   * @param Mzz Mzz component of moment tensor
   * @param Mxz Mxz component of moment tensor
   * @param forcing_function pointer to source time function
   * @param wavefield_type type of wavefield
   *
   */
  moment_tensor(
      type_real x, type_real z, const type_real Mxx, const type_real Mzz,
      const type_real Mxz,
      std::unique_ptr<specfem::forcing_function::stf> forcing_function,
      const specfem::wavefield::simulation_field wavefield_type)
      : Mxx(Mxx), Mzz(Mzz), Mxz(Mxz), wavefield_type(wavefield_type),
        tensor_source<specfem::dimension::type::dim2>(
            x, z, std::move(forcing_function)) {};

  /**
   * @brief User output
   *
   */
  std::string print() const override;

  specfem::wavefield::simulation_field get_wavefield_type() const override {
    return wavefield_type;
  }

  bool operator==(const specfem::sources::source<specfem::dimension::type::dim2>
                      &other) const override;
  bool operator!=(const specfem::sources::source<specfem::dimension::type::dim2>
                      &other) const override;

  /**
   * @brief Get the source tensor
   *
   * @return Kokkos::View<type_real **, Kokkos::LayoutLeft, Kokkos::HostSpace>
   * Source tensor with dimensions [ncomponents][2] where each row contains
   * [Mxx, Mxz], [Mxz, Mzz] etc, depending on the medium type
   */
  specfem::kokkos::HostView2d<type_real> get_source_tensor() const override;

  /**
   * @brief Get the list of supported media for this source type
   *
   * @return std::vector<specfem::element::medium_tag> list of supported media
   */
  std::vector<specfem::element::medium_tag>
  get_supported_media() const override;

private:
  type_real Mxx;                                       ///< Mxx for the source
  type_real Mxz;                                       ///< Mxz for the source
  type_real Mzz;                                       ///< Mzz for the source
  specfem::wavefield::simulation_field wavefield_type; ///< Type of wavefield on
                                                       ///< which the source
                                                       ///< acts

public:
  static constexpr const char *name = "2-D moment tensor";

protected:
};
} // namespace sources
} // namespace specfem
