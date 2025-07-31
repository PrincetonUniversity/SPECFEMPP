#pragma once

#include "specfem/source.hpp"

namespace specfem {
namespace sources {

template <>
class tensor_source<specfem::dimension::type::dim3>
    : public source<specfem::dimension::type::dim3> {

public:
  /**
   * @brief Default tensor source constructor
   *
   */
  tensor_source() {};

  /**
   * @brief Construct a new tensor source object using the forcing function
   *
   * @param x x-coordinate of source
   * @param y y-coordinate of source
   * @param z z-coordinate of source
   * @param forcing_function pointer to source time function
   * @param wavefield_type type of wavefield
   */
  tensor_source(
      type_real x, type_real y, type_real z,
      std::unique_ptr<specfem::forcing_function::stf> forcing_function)
      : source(x, y, z, std::move(forcing_function)) {};

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
   * Source tensor with dimensions [ncomponents][3] where each row contains
   * [Mxx, Mxy, Mxz], [Mxy, Myy, Myz], [Mxz, Myz, Mzz] etc, depending on the
   * medium type
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
