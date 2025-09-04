#pragma once

#include "enumerations/interface.hpp"
#include "kokkos_abstractions.h"
#include "specfem/source.hpp"
#include "yaml-cpp/yaml.h"

namespace specfem {
namespace sources {

/**
 * @brief Class representing a tensor source
 *
 * The tensor source class is a base class for all tensor sources in the
 * simulation. It provides the common interface and functionality for
 * manipulating tensor sources. The main functionality being the return of a
 * tensor that can be used to compute the GLL level source array, which is
 * applied in the simulation.
 *
 * The main differences between 2D and 3D tensor sources are the dimensions and
 * global and local coordinates for the point sources.
 *
 * @tparam DimensionTag Dimension of the tensor source
 */
template <specfem::dimension::type DimensionTag>
class tensor_source : public source<DimensionTag> {

public:
  /**
   * @brief Default tensor source constructor
   *
   */
  tensor_source() {};

  /**
   * @brief Construct a new 2D tensor source object
   *
   * @param x x-coordinate of source
   * @param z z-coordinate of source
   * @param forcing_function pointer to source time function
   */
  template <specfem::dimension::type U = DimensionTag,
            typename std::enable_if<U == specfem::dimension::type::dim2>::type
                * = nullptr>
  tensor_source(
      type_real x, type_real z,
      std::unique_ptr<specfem::forcing_function::stf> forcing_function)
      : source<DimensionTag>(x, z, std::move(forcing_function)){};

  /**
   * @brief Construct a new 3D tensor source object
   * @param x x-coordinate of source
   * @param y y-coordinate of source
   * @param z z-coordinate of source
   * @param forcing_function pointer to source time function
   */
  template <specfem::dimension::type U = DimensionTag,
            typename std::enable_if<U == specfem::dimension::type::dim3>::type
                * = nullptr>
  tensor_source(
      type_real x, type_real y, type_real z,
      std::unique_ptr<specfem::forcing_function::stf> forcing_function)
      : source<DimensionTag>(x, y, z, std::move(forcing_function)){};

  /**
   * @brief Construct a new tensor source object from a YAML node and time steps
   *
   * @param Node YAML node defining the tensor source
   * @param nsteps Number of time steps in the simulation
   * @param dt Time increment in the simulation
   * @param wavefield_type Type of wavefield on which the source acts
   */
  tensor_source(YAML::Node &Node, const int nsteps, const type_real dt)
      : source<DimensionTag>(Node, nsteps, dt) {};

  /**
   * @brief Get the source tensor
   *
   * @return Kokkos::View<type_real **, Kokkos::LayoutLeft, Kokkos::HostSpace>
   * Source tensor with dimensions [ncomponents][2] where -- in the case of the
   * moment tensor, each row contains [Mxx, Mxz], [Mxz, Mzz] etc, depending on
   * the medium type
   *
   * or in 3D
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
