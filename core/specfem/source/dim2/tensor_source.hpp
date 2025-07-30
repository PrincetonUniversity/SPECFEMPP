#pragma once

#include "enumerations/interface.hpp"
#include "kokkos_abstractions.h"
#include "specfem/source.hpp"
#include "yaml-cpp/yaml.h"

namespace specfem {
namespace sources {

template <>
class tensor_source<specfem::dimension::type::dim2>
    : public source<specfem::dimension::type::dim2> {

public:
  /**
   * @brief Default tensor source constructor
   *
   */
  tensor_source() {};

  tensor_source(
      type_real x, type_real z,
      std::unique_ptr<specfem::forcing_function::stf> forcing_function)
      : source(x, z, std::move(forcing_function)) {};

  /**
   * @brief Construct a new tensor source object from a YAML node and time steps
   *
   * @param Node YAML node defining the tensor source
   * @param nsteps Number of time steps in the simulation
   * @param dt Time increment in the simulation
   * @param wavefield_type Type of wavefield on which the source acts
   */
  tensor_source(YAML::Node &Node, const int nsteps, const type_real dt)
      : source(Node, nsteps, dt) {};

  /**
   * @brief Get the source tensor
   *
   * @return Kokkos::View<type_real **, Kokkos::LayoutLeft, Kokkos::HostSpace>
   * Source tensor with dimensions [ncomponents][2] where each row contains
   * [Mxx, Mxz], [Mxz, Mzz] etc, depending on the medium type
   */
  virtual specfem::kokkos::HostView2d<type_real> get_source_tensor() const = 0;

  /**
   * @brief Get the source type
   *
   * @return source_type type of source
   */
  source_type get_source_type() const override {
    return source_type::tensor_source;
  }
};
} // namespace sources
} // namespace specfem
