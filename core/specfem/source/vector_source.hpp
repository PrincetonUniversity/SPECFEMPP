#pragma once

#include "enumerations/interface.hpp"
#include "specfem/source.hpp"

namespace specfem {
namespace sources {

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
   * @return Kokkos::View<type_real *, Kokkos::LayoutLeft, Kokkos::HostSpace>
   * Force vector
   */
  virtual specfem::kokkos::HostView1d<type_real> get_force_vector() const = 0;
};

} // namespace sources
} // namespace specfem
