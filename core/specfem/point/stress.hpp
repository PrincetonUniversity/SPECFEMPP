#pragma once

/**
 * @file stress.hpp
 * @brief Stress tensor data container for spectral element continuum mechanics.
 *
 * This file provides specialized stress tensor containers for wave propagation
 * and solid mechanics simulations. Stress tensors represent the internal forces
 * per unit area within continuous media and are fundamental to constitutive
 * relations in continuum mechanics.
 *
 * **Physical Context:**
 * The Cauchy stress tensor \f$\boldsymbol{\sigma}\f$ describes internal forces
 * in deformed materials. For a material element with normal vector
 * \f$\mathbf{n}\f$, the stress vector is given by:
 * \f$
 *   \mathbf{t} = \boldsymbol{\sigma} \mathbf{n}
 * \f$
 *
 * **Tensor Structure:**
 * - **2D**: \f$\boldsymbol{\sigma} = \begin{bmatrix} \sigma_{xx} & \sigma_{xz}
 * \\ \sigma_{zx} & \sigma_{zz} \end{bmatrix}\f$
 * - **3D**: Full symmetric tensor with 6 independent components
 * - **Acoustic**: Scalar pressure \f$p =
 * -\frac{1}{3}\text{tr}(\boldsymbol{\sigma})\f$
 *
 * **Constitutive Relations:**
 * Stress relates to strain through material properties:
 * - **Linear elastic**: \f$\boldsymbol{\sigma} = \mathbf{C} :
 * \boldsymbol{\varepsilon}\f$
 * - **Nonlinear**: Various hyperelastic and plastic models
 * - **Viscoelastic**: Time-dependent stress-strain relations
 *
 */

#include "datatypes/point_view.hpp"
#include "enumerations/interface.hpp"
#include "jacobian_matrix.hpp"
#include "specfem/data_access.hpp"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace point {

/**
 * @brief Stress tensor representation for spectral element continuum mechanics
 * simulations.
 *
 * The stress class provides a comprehensive interface for manipulating stress
 * tensor data at individual quadrature points within spectral elements. It
 * encapsulates the mathematical structure of the Cauchy stress tensor while
 * providing physics-aware operations for stress transformations and
 * constitutive computations.
 *
 * **Mathematical Foundation:**
 * The Cauchy stress tensor \f$\boldsymbol{\sigma}(\mathbf{x}, t)\f$ is a
 * second-order tensor that relates internal forces to surface area elements.
 * For an infinitesimal area element with normal vector \f$\mathbf{n}\f$, the
 * stress vector (force per unit area) is:
 * \f$
 *   \mathbf{t} = \boldsymbol{\sigma} \mathbf{n}
 * \f$
 *
 * **Tensor Components:**
 * The stress tensor structure depends on the medium type and spatial dimension:
 *
 * - **Acoustic medium (2D/3D)**:
 *   - 1 component: scalar pressure \f$p\f$
 *   - Relation: \f$\boldsymbol{\sigma} = -p \mathbf{I}\f$ (isotropic pressure)
 *
 * - **Elastic medium (2D)**:
 *   - 3 independent components: \f$\sigma_{xx}, \sigma_{zz}, \sigma_{xz}\f$
 *   - Tensor: \f$\boldsymbol{\sigma} = \begin{bmatrix} \sigma_{xx} &
 * \sigma_{xz} \\ \sigma_{xz} & \sigma_{zz} \end{bmatrix}\f$
 *
 * - **Elastic medium (3D)**:
 *   - 6 independent components due to symmetry
 *   - Full tensor: \f$\boldsymbol{\sigma} = \begin{bmatrix} \sigma_{xx} &
 * \sigma_{xy} & \sigma_{xz} \\ \sigma_{xy} & \sigma_{yy} & \sigma_{yz}
 * \\ \sigma_{xz} & \sigma_{yz} & \sigma_{zz} \end{bmatrix}\f$
 *
 * - **Poroelastic medium**:
 *   - Separate stress tensors for solid and fluid phases
 *   - Effective stress principle in saturated porous media
 *
 * **Physical Interpretation:**
 * - **Normal stresses**: \f$\sigma_{ii}\f$ represent compression (negative) or
 * tension (positive)
 * - **Shear stresses**: \f$\sigma_{ij} (i \neq j)\f$ represent tangential
 * forces
 * - **Hydrostatic pressure**: \f$p =
 * -\frac{1}{3}\text{tr}(\boldsymbol{\sigma})\f$
 * - **Deviatoric stress**: \f$\mathbf{s} = \boldsymbol{\sigma} + p\mathbf{I}\f$
 *
 * **Stress Transformations:**
 * The class provides operations for coordinate transformations essential in
 * spectral elements:
 * - **Reference to physical**: Transform stress from computational to physical
 * coordinates
 * - **Contravariant mapping**: Handle non-orthogonal coordinate systems
 * - **Jacobian weighting**: Account for element geometry in integration
 *
 * **Constitutive Integration:**
 * Stress computation from strain through material laws:
 * \f$
 *   \boldsymbol{\sigma}^{n+1} = f(\boldsymbol{\varepsilon}^{n+1},
 * \boldsymbol{\sigma}^n, \text{history})
 * \f$
 * where \f$f\f$ represents the constitutive model (linear elastic, plasticity,
 * etc.)
 *
 * @tparam DimensionTag Spatial dimension determining tensor structure:
 *                      - `specfem::dimension::type::dim2`: 2D plane
 * stress/strain
 *                      - `specfem::dimension::type::dim3`: 3D full tensor
 *
 * @tparam MediumTag Physical medium type defining stress behavior:
 *                   - `specfem::element::medium_tag::acoustic`: Scalar pressure
 *                   - `specfem::element::medium_tag::elastic`: Full stress
 * tensor
 *                   - `specfem::element::medium_tag::poroelastic`: Coupled
 * solid-fluid stresses
 *
 * @tparam UseSIMD Boolean flag for SIMD vectorization enabling simultaneous
 *                 stress operations across multiple quadrature points.
 *
 * @note The stress class integrates with the SPECFEMPP data access framework
 *       providing SIMD vectorization and efficient memory access patterns
 *       essential for high-performance continuum mechanics computations.
 *
 * @see specfem::point::jacobian_matrix
 * @see specfem::constitutive_laws
 * @see specfem::data_access::Accessor
 * @see specfem::point::field_derivatives
 *
 * @code
 * // Example: Creating 2D elastic stress tensor
 * using StressType = specfem::point::stress<
 *     specfem::dimension::type::dim2,
 *     specfem::element::medium_tag::elastic,
 *     false>;
 *
 * // Initialize stress components for plane stress (sigma_xx, sigma_zz,
 * sigma_xz) typename StressType::value_type stress_components( 1000.0,  //
 * sigma_xx (Pa) - normal stress in x-direction -500.0,  // sigma_zz (Pa) -
 * normal stress in z-direction (compression) 200.0    // sigma_xz (Pa) - shear
 * stress
 * );
 * StressType stress_tensor(stress_components);
 *
 * // Compute stress invariants
 * auto hydrostatic_pressure = (stress_components[0] + stress_components[1])
 * / 2.0; auto max_shear = std::abs(stress_components[2]);
 * @endcode
 *
 * @code
 * // Example: Stress transformation with Jacobian matrix
 * auto jacobian = specfem::point::jacobian_matrix<dim2, false>();
 *
 * // Transform stress from reference to physical coordinates
 * auto physical_stress = stress_tensor * jacobian;
 *
 * // This transformation accounts for:
 * // 1. Coordinate system rotation
 * // 2. Element deformation mapping
 * // 3. Metric tensor effects in curvilinear coordinates
 * @endcode
 *
 * @code
 * // Example: Linear elastic constitutive relation
 * using StrainType = specfem::point::field_derivatives<dim2, elastic, false>;
 *
 * StrainType strain_tensor;
 * StressType stress_tensor;
 *
 * // Material properties (isotropic elasticity)
 * const auto lambda = 1.0e9;  // First Lamé parameter (Pa)
 * const auto mu = 8.0e8;      // Shear modulus (Pa)
 *
 * // Linear elastic stress-strain relation
 * stress_tensor(0) = (lambda + 2*mu) * strain_tensor.eps_xx() + lambda *
 * strain_tensor.eps_zz(); stress_tensor(1) = lambda * strain_tensor.eps_xx() +
 * (lambda + 2*mu) * strain_tensor.eps_zz(); stress_tensor(2) = 2*mu *
 * strain_tensor.eps_xz();
 * @endcode
 *
 * @code
 * // Example: Acoustic pressure stress
 * using AcousticStress = specfem::point::stress<
 *     specfem::dimension::type::dim2,
 *     specfem::element::medium_tag::acoustic,
 *     true>;  // SIMD enabled for performance
 *
 * AcousticStress pressure;
 *
 * // In acoustic media, stress is simply negative pressure
 * auto bulk_modulus = 2.2e9;  // Pa (water)
 * auto volumetric_strain = compute_divergence(velocity_field);
 * pressure(0) = -bulk_modulus * volumetric_strain;
 * @endcode
 *
 * @code
 * // Example: Integration with assembly system
 * StressType stress_tensor;
 * auto jacobian_matrix = element.get_jacobian_matrix(quadrature_point);
 * auto quadrature_weight = element.get_quadrature_weight(quadrature_point);
 *
 * // Transform and integrate stress contribution
 * auto transformed_stress = stress_tensor * jacobian_matrix;
 * auto stress_contribution = transformed_stress * jacobian_matrix.determinant()
 * * quadrature_weight;
 *
 * // Add to global force vector
 * assembly_system.add_stress_contribution(element_dofs, stress_contribution);
 * @endcode
 */
template <specfem::dimension::type DimensionTag,
          specfem::element::medium_tag MediumTag, bool UseSIMD>
struct stress
    : public specfem::data_access::Accessor<
          specfem::data_access::AccessorType::point,
          specfem::data_access::DataClassType::stress, DimensionTag, UseSIMD> {
private:
  /** @brief Base accessor type for data access framework integration */
  using base_type = specfem::data_access::Accessor<
      specfem::data_access::AccessorType::point,
      specfem::data_access::DataClassType::stress, DimensionTag, UseSIMD>;

public:
  /**
   * @name Static Properties
   * @brief Compile-time constants derived from template parameters
   */
  ///@{
  /** @brief Spatial dimension (2 or 3) */
  constexpr static int dimension =
      specfem::element::attributes<DimensionTag, MediumTag>::dimension;

  /** @brief Number of stress components based on medium type */
  constexpr static int components =
      specfem::element::attributes<DimensionTag, MediumTag>::components;

  /** @brief Template parameter for spatial dimension */
  constexpr static specfem::dimension::type dimension_tag = DimensionTag;

  /** @brief Template parameter for medium type */
  constexpr static specfem::element::medium_tag medium_tag = MediumTag;

  /** @brief Template parameter for SIMD usage */
  constexpr static bool using_simd = UseSIMD;
  ///@}

  /**
   * @name Type Definitions
   * @brief Type aliases for SIMD and tensor operations
   */
  ///@{
  /** @brief SIMD type for vectorized operations */
  using simd = typename base_type::template simd<type_real>;

  /** @brief Tensor type for storing stress components (components × dimension)
   */
  using value_type =
      typename base_type::template tensor_type<type_real, components,
                                               dimension>;
  ///@}

  /**
   * @name Data Members
   */
  ///@{
  /** @brief Stress tensor storage with shape (components × dimension) */
  value_type T;
  ///@}

  /**
   * @name Constructors
   */
  ///@{
  /**
   * @brief Default constructor.
   *
   * Initializes stress tensor with default values (typically zero).
   */
  KOKKOS_FUNCTION stress() = default;

  /**
   * @brief Constructor with stress tensor initialization.
   *
   * @param T Stress tensor with components arranged as (components × dimension)
   *
   * @code
   * // For 2D elastic medium (2 components × 2 dimensions)
   * typename stress_type::value_type tensor(1.1, 2.1,  // component 0: (σxx,
   * σxz) 1.2, 2.2); // component 1: (σzx, σzz) stress_type stress(tensor);
   * @endcode
   */
  KOKKOS_FUNCTION stress(const value_type &T) : T(T) {}
  ///@}

  /**
   * @name Operators
   */
  ///@{
  /**
   * @brief Transform stress tensor using jacobian matrix.
   *
   * Applies the coordinate transformation from reference element to physical
   * element using the jacobian matrix. This operation transforms stress
   * components from the reference (ξ, ζ) coordinate system to the physical (x,
   * z) coordinate system.
   *
   * The transformation formula for 2D is:
   * \f$ F(i,0) = J \cdot (T(i,0) \cdot \frac{\partial\xi}{\partial x} + T(i,1)
   * \cdot \frac{\partial\zeta}{\partial x}) \f$
   * \f$ F(i,1) = J \cdot (T(i,0) \cdot \frac{\partial\xi}{\partial z} + T(i,1)
   * \cdot \frac{\partial\zeta}{\partial z}) \f$
   *
   * where \f$ J \f$ is the jacobian determinant and the partial derivatives are
   * the inverse jacobian matrix elements.
   *
   * @param jacobian_matrix Jacobian matrix containing transformation
   * derivatives
   * @return Transformed stress tensor in physical coordinates
   *
   * @code
   * stress_type stress(stress_tensor);
   * auto jacobian = compute_jacobian_matrix(quadrature_point);
   * auto transformed = stress * jacobian;
   * @endcode
   */
  KOKKOS_INLINE_FUNCTION
  value_type operator*(
      const specfem::point::jacobian_matrix<specfem::dimension::type::dim2,
                                            true, UseSIMD> &jacobian_matrix)
      const {
    value_type F;

    for (int icomponent = 0; icomponent < components; ++icomponent) {
      F(icomponent, 0) =
          jacobian_matrix.jacobian * (T(icomponent, 0) * jacobian_matrix.xix +
                                      T(icomponent, 1) * jacobian_matrix.xiz);
      F(icomponent, 1) = jacobian_matrix.jacobian *
                         (T(icomponent, 0) * jacobian_matrix.gammax +
                          T(icomponent, 1) * jacobian_matrix.gammaz);
    }

    return F;
  }

  /**
   * @brief Equality comparison operator.
   *
   * Compares two stress tensors for equality by comparing their underlying
   * tensor data element-wise.
   *
   * @param other Another stress tensor to compare with
   * @return true if tensors are equal, false otherwise
   */
  KOKKOS_INLINE_FUNCTION
  bool operator==(const stress &other) const { return T == other.T; };
  ///@}

  /**
   * @name Utility Functions
   */
  ///@{
  /**
   * @brief Generate string representation of the stress tensor.
   *
   * Creates a formatted string showing all components of the stress tensor
   * for debugging and visualization purposes. The output format shows each
   * component with its (component, dimension) indices.
   *
   * @return Formatted string representation of the stress tensor
   *
   * @code
   * stress_type stress(tensor_data);
   * std::cout << stress.print() << std::endl;
   * // Output:
   * // Stress Tensor:
   * // T(0, 0) = 1.1, T(0, 1) = 1.2
   * // T(1, 0) = 2.1, T(1, 1) = 2.2
   * @endcode
   */
  std::string print() const {
    std::ostringstream oss;
    oss << "Stress Tensor:\n";
    for (int i = 0; i < components; ++i) {
      oss << "T(" << i << ", 0) = " << T(i, 0) << ", "
          << "T(" << i << ", 1) = " << T(i, 1) << "\n";
    }
    return oss.str();
  }
  ///@}
};
} // namespace point
} // namespace specfem
