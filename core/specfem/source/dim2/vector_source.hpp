#pragma once

#include "enumerations/interface.hpp"
#include "specfem/source.hpp"

namespace specfem {
namespace sources {

template <>
class vector_source<specfem::dimension::type::dim2>
    : public source<specfem::dimension::type::dim2> {

public:
  /**
   * @brief Default vector source constructor
   *
   */
  vector_source() {};

  /**
   * @brief Construct a new vector source object using the forcing function
   *
   * @param x x-coordinate of source
   * @param y z-coordinate of source
   * @param forcing_function pointer to source time function
   * @param wavefield_type type of wavefield
   */
  vector_source(
      type_real x, type_real y,
      std::unique_ptr<specfem::forcing_function::stf> forcing_function)
      : source(x, y, std::move(forcing_function)) {};

  /**
   * @brief Construct a new vector source object from a YAML node and time steps
   *
   * @param Node YAML node defining the vector source
   * @param nsteps Number of time steps in the simulation
   * @param dt Time increment in the simulation
   * @param wavefield_type Type of wavefield on which the source acts
   */
  vector_source(YAML::Node &Node, const int nsteps, const type_real dt)
      : source(Node, nsteps, dt) {};

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
   * @return Kokkos::View<type_real *, Kokkos::LayoutLeft, Kokkos::HostSpace>
   * Force vector
   */
  virtual specfem::kokkos::HostView1d<type_real> get_force_vector() const = 0;
};

} // namespace sources
} // namespace specfem
