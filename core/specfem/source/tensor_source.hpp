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
 * Tensor sources represent stress or moment tensor sources, typically used
 * for earthquake simulations. They provide a symmetric tensor representation
 * of the source mechanism.
 *
 * The main differences between 2D and 3D tensor sources are the dimensions and
 * global and local coordinates for the point sources.
 *
 * @tparam DimensionTag Dimension of the tensor source (dim2 or dim3)
 *
 * @par Examples of tensor sources:
 * - @ref specfem::sources::moment_tensor - Seismic moment tensor sources
 *
 * @par Tensor Source Usage Pattern
 * @code
 * // Example: Creating and using a tensor source (2D moment tensor)
 * auto stf = std::make_unique<specfem::source_time_functions::Ricker>(
 *     8.0, 0.01, 1.0, 0.0, 1.0, false
 * );
 *
 * auto tensor_src =
 * specfem::sources::moment_tensor<specfem::dimension::type::dim2>( 12.5, 8.0,
 * // coordinates (x, z) 1.5,        // Mxx - normal stress in x direction 2.1,
 * // Mzz - normal stress in z direction 0.7,        // Mxz - shear stress
 * component std::move(stf), specfem::wavefield::simulation_field::forward
 * );
 *
 * // Set the medium (tensor sources work with elastic media)
 * tensor_src.set_medium_tag(specfem::element::medium_tag::elastic_psv);
 *
 * // Get the source tensor - dimensions depend on simulation:
 * // - 2D: 2x2 symmetric matrix [[Mxx, Mxz], [Mxz, Mzz]]
 * // - 3D: 3x3 symmetric matrix with 6 independent components
 * auto source_tensor = tensor_src.get_source_tensor();
 *
 * // All tensor sources return tensor_source type
 * assert(tensor_src.get_source_type() ==
 *        specfem::sources::source_type::tensor_source);
 *
 * // Access individual tensor components (for moment tensors)
 * type_real mxx = tensor_src.get_Mxx();
 * type_real mzz = tensor_src.get_Mzz();
 * type_real mxz = tensor_src.get_Mxz();
 * @endcode
 *
 * @note This class inherits from @ref specfem::sources::source
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
   * @param source_time_function pointer to source time function
   */
  template <specfem::dimension::type U = DimensionTag,
            typename std::enable_if<U == specfem::dimension::type::dim2>::type
                * = nullptr>
  tensor_source(
      type_real x, type_real z,
      std::unique_ptr<specfem::source_time_functions::stf> source_time_function)
      : source<DimensionTag>(x, z, std::move(source_time_function)){};

  /**
   * @brief Construct a new 3D tensor source object
   * @param x x-coordinate of source
   * @param y y-coordinate of source
   * @param z z-coordinate of source
   * @param source_time_function pointer to source time function
   */
  template <specfem::dimension::type U = DimensionTag,
            typename std::enable_if<U == specfem::dimension::type::dim3>::type
                * = nullptr>
  tensor_source(
      type_real x, type_real y, type_real z,
      std::unique_ptr<specfem::source_time_functions::stf> source_time_function)
      : source<DimensionTag>(x, y, z, std::move(source_time_function)){};

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
   * Returns the source tensor \f$\mathbf{M}\f$ representing the stress or
   * moment tensor applied by this tensor source. The tensor is symmetric and
   * its dimensionality depends on the simulation dimension and medium type.
   *
   * @par Mathematical Definition
   * The source tensor represents the symmetric stress/moment tensor:
   *
   * **2D Case:**
   * \f[
   * \mathbf{M}_{2D} = \begin{pmatrix}
   * M_{xx} & M_{xz} \\
   * M_{xz} & M_{zz}
   * \end{pmatrix}
   * \f]
   *
   * **3D Case:**
   * \f[
   * \mathbf{M}_{3D} = \begin{pmatrix}
   * M_{xx} & M_{xy} & M_{xz} \\
   * M_{xy} & M_{yy} & M_{yz} \\
   * M_{xz} & M_{yz} & M_{zz}
   * \end{pmatrix}
   * \f]
   *
   * The returned tensor format depends on the medium:
   * - **Elastic PSV (2D)**: 2×2 matrix \f$[[M_{xx}, M_{xz}], [M_{xz},
   * M_{zz}]]\f$
   * - **Elastic PSV-T (2D)**: 3×2 matrix with additional rotational component
   * - **Poroelastic (2D)**: 4×2 matrix (elastic tensor repeated for
   * solid/fluid)
   * - **Elastic (3D)**: 3×3 full moment tensor
   * - **Electromagnetic**: Modified tensor for Maxwell equations
   *
   * where \f$M_{ij}\f$ are the independent components of the symmetric tensor,
   * representing seismic moment, stress, or equivalent source mechanisms.
   *
   * @note The actual tensor values and their physical meaning depend on the
   * specific source type implementation. See individual source classes for
   * detailed mathematical definitions of their tensor components.
   *
   * @return Kokkos::View<type_real **, Kokkos::LayoutLeft, Kokkos::HostSpace>
   * Source tensor matrix with dimensions [ncomponents][ndim] where ncomponents
   * depends on medium type and ndim is spatial dimension
   */
  virtual Kokkos::View<type_real **, Kokkos::LayoutRight, Kokkos::HostSpace>
  get_source_tensor() const = 0;

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
