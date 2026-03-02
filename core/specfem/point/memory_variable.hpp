#include "specfem/constants.hpp"
#include "specfem/data_access/accessor.hpp"
#include "specfem/datatype.hpp"

namespace specfem::point {

/**
 * @brief Template class for storing memory variables in attenuation
 * calculations.
 *
 * The memory_variable class stores stress tensor components used in
 * viscoelastic attenuation modeling for seismic wave simulation. It supports
 * both 2D and 3D spatial dimensions with specialized implementations for
 * elastic media with constant isotropic attenuation.
 *
 * @tparam DimensionTag Spatial dimension (dim2 or dim3)
 * @tparam MediumTag Medium type (currently supports elastic)
 * @tparam AttenuationTag Attenuation model (constant_isotropic)
 * @tparam UseSIMD Enable SIMD vectorization optimizations
 *
 * The class integrates with the SPECFEM++ data access framework through
 * inheritance from specfem::data_access::Accessor, providing efficient
 * memory management and SIMD support for high-performance computing.
 *
 * @code
 * // Example usage for 2D elastic medium:
 * using mv_2d = specfem::point::memory_variable<
 *     specfem::element::dimension_tag::dim2,
 *     specfem::element::medium_tag::elastic,
 *     specfem::element::attenuation_tag::constant_isotropic,
 *     false>;
 *
 * mv_2d memory_vars;
 * memory_vars.Rxx = mv_2d::value_type(1.5);
 * memory_vars.Rxz = mv_2d::value_type(0.8);
 * @endcode
 */
template <specfem::element::dimension_tag DimensionTag,
          specfem::element::medium_tag MediumTag,
          specfem::element::attenuation_tag AttenuationTag, bool UseSIMD>
struct memory_variable;

/**
 * @brief Specialized memory_variable for 2D elastic medium with constant
 * isotropic attenuation.
 *
 * This specialization stores the memory variables required for 2D viscoelastic
 * attenuation calculations. In 2D, the stress tensor requires three independent
 * components: Rxx, Rxz, and Rkappa. The Rzz component is computed from Rkappa
 * and Rxx.
 *
 * The class provides arithmetic operators for memory variable updates during
 * time stepping in seismic wave propagation simulations.
 *
 * @tparam UseSIMD Enable SIMD vectorization for performance optimization
 */
template <bool UseSIMD>
struct memory_variable<specfem::element::dimension_tag::dim2,
                       specfem::element::medium_tag::elastic,
                       specfem::element::attenuation_tag::constant_isotropic,
                       UseSIMD>
    : public specfem::data_access::Accessor<
          specfem::datatype::AccessorType::point,
          specfem::data_access::DataClassType::memory_variable,
          specfem::element::dimension_tag::dim2, UseSIMD> {
private:
  using base_type = specfem::data_access::Accessor<
      specfem::datatype::AccessorType::point,
      specfem::data_access::DataClassType::memory_variable,
      specfem::element::dimension_tag::dim2, UseSIMD>;

public:
  /**
   * @name Typedefs
   *
   */
  ///@{
  using simd = typename base_type::template simd<type_real>; ///< SIMD data type
  using value_type =
      typename base_type::template vector_type<type_real,
                                               specfem::constants::N_SLS>;
  ///@}

  /**
   * @name Memory Variable Components
   * @brief Stress tensor memory variables for 2D attenuation calculations
   */
  ///@{
  value_type Rxx; ///< Memory variable for \f$R_{xx}\f$ stress component
  value_type Rxz; ///< Memory variable for \f$R_{xz}\f$ shear stress component
  value_type Rkappa; ///< Memory variable for bulk modulus component,
                     ///< \f$R_{zz}\f$ computed from \f$R_{\kappa}\f$ and
                     ///< \f$R_{xx}\f$
  ///@}

  /**
   * @name Constructors
   *
   */
  ///@{
  /**
   * @brief Default constructor
   *
   * Initializes all memory variable components to zero using the init() method.
   * This constructor is suitable for creating empty memory variable objects
   * that will be populated later during simulation initialization.
   */
  KOKKOS_FUNCTION
  memory_variable() {
    this->init();
    return;
  }

  /**
   * @brief Component-wise constructor
   *
   * Directly initializes memory variables with specified values for each
   * stress tensor component.
   *
   * @param Rxx Memory variable for \f$R_{xx}\f$ component
   * @param Rxz Memory variable for \f$R_{xz}\f$ component
   * @param Rkappa Memory variable for \f$R_{\kappa}\f$ bulk modulus component
   */
  KOKKOS_FUNCTION
  memory_variable(const value_type &Rxx, const value_type &Rxz,
                  const value_type &Rkappa)
      : Rxx(Rxx), Rxz(Rxz), Rkappa(Rkappa) {}

  /**
   * @brief Uniform initialization constructor
   *
   * Initializes all memory variable components to the same value.
   * Useful for testing or when all components start with identical values.
   *
   * @param constant Value to initialize all memory variable components
   *
   * @code
   * memory_variable<...> mv(mv_type::value_type(1.0));
   * // Results in: Rxx = Rxz = Rkappa = 1.0
   * @endcode
   */
  KOKKOS_FUNCTION
  memory_variable(const value_type constant)
      : Rxx(constant), Rxz(constant), Rkappa(constant) {}

  /**
   * @brief Initialize all memory variables to zero
   *
   * Sets all component arrays (Rxx, Rxz, Rkappa) to zero values.
   * Used internally by the default constructor and can be called
   * to reset memory variables during simulation.
   */
  KOKKOS_FUNCTION
  void init() {
    this->Rxx = value_type(typename value_type::value_type(0));
    this->Rxz = value_type(typename value_type::value_type(0));
    this->Rkappa = value_type(typename value_type::value_type(0));
    return;
  }

  /**
   * @name Arithmetic Operators
   * @brief Mathematical operations for memory variable updates
   */
  ///@{

  /**
   * @brief Addition operator for memory variables
   *
   * Performs component-wise addition across all N_SLS standard linear solids.
   * Used in attenuation update calculations during time stepping.
   *
   * @param rhs Right-hand side memory variable
   * @return New memory variable containing the sum
   */
  KOKKOS_FUNCTION memory_variable operator+(const memory_variable &rhs) const {
    memory_variable result;
    constexpr int N = specfem::constants::N_SLS;
    for (int i = 0; i < N; ++i) {
      result.Rxx(i) = this->Rxx(i) + rhs.Rxx(i);
      result.Rxz(i) = this->Rxz(i) + rhs.Rxz(i);
      result.Rkappa(i) = this->Rkappa(i) + rhs.Rkappa(i);
    }
    return result;
  }

  /**
   * @brief In-place addition operator
   *
   * Adds another memory variable to this one, modifying this object.
   * Efficient for accumulating contributions during simulation.
   *
   * @param rhs Memory variable to add
   * @return Reference to this modified object
   */
  KOKKOS_FUNCTION memory_variable &operator+=(const memory_variable &rhs) {
    constexpr int N = specfem::constants::N_SLS;
    for (int i = 0; i < N; ++i) {
      this->Rxx(i) += rhs.Rxx(i);
      this->Rxz(i) += rhs.Rxz(i);
      this->Rkappa(i) += rhs.Rkappa(i);
    }
    return *this;
  }

  /**
   * @brief Scalar multiplication operator
   *
   * Multiplies all memory variable components by a scalar value.
   * Used for scaling operations in attenuation calculations.
   *
   * @param rhs Scalar multiplier
   * @return New memory variable with scaled components
   */
  KOKKOS_FUNCTION memory_variable operator*(const type_real &rhs) const {
    memory_variable result;
    constexpr int N = specfem::constants::N_SLS;
    for (int i = 0; i < N; ++i) {
      result.Rxx(i) = this->Rxx(i) * rhs;
      result.Rxz(i) = this->Rxz(i) * rhs;
      result.Rkappa(i) = this->Rkappa(i) * rhs;
    }
    return result;
  }

  /**
   * @brief Equality comparison operator
   *
   * Compares two memory variables for exact equality across all components.
   * Used primarily in unit testing and validation.
   *
   * @param rhs Memory variable to compare with
   * @return true if all components are equal, false otherwise
   */
  KOKKOS_FUNCTION bool operator==(const memory_variable &rhs) const {
    return (this->Rxx == rhs.Rxx) && (this->Rxz == rhs.Rxz) &&
           (this->Rkappa == rhs.Rkappa);
  }
  ///@}
};

/**
 * @brief Specialized memory_variable for 3D elastic medium with constant
 * isotropic attenuation.
 *
 * This specialization handles memory variables for 3D viscoelastic attenuation.
 * In 3D, the stress tensor requires six independent components: Rxx, Ryy, Rxy,
 * Rxz, Ryz, and Rkappa. The Rzz component is derived from Rkappa, Rxx, and Ryy.
 *
 * This class supports the full 3D stress tensor operations needed for accurate
 * seismic wave simulation with attenuation in three-dimensional media.
 *
 * @tparam UseSIMD Enable SIMD vectorization for performance optimization
 */
template <bool UseSIMD>
struct memory_variable<specfem::element::dimension_tag::dim3,
                       specfem::element::medium_tag::elastic,
                       specfem::element::attenuation_tag::constant_isotropic,
                       UseSIMD>
    : public specfem::data_access::Accessor<
          specfem::datatype::AccessorType::point,
          specfem::data_access::DataClassType::memory_variable,
          specfem::element::dimension_tag::dim3, UseSIMD> {
private:
  using base_type = specfem::data_access::Accessor<
      specfem::datatype::AccessorType::point,
      specfem::data_access::DataClassType::memory_variable,
      specfem::element::dimension_tag::dim3, UseSIMD>;

public:
  /**
   * @name Typedefs
   *
   */
  ///@{
  using simd = typename base_type::template simd<type_real>; ///< SIMD data type
  using value_type =
      typename base_type::template vector_type<type_real,
                                               specfem::constants::N_SLS>;
  ///@}

  /**
   * @name Memory Variable Components
   * @brief Stress tensor memory variables for 3D attenuation calculations
   */
  ///@{
  value_type Rxx; ///< Memory variable for \f$R_{xx}\f$ normal stress component
  value_type Ryy; ///< Memory variable for \f$R_{yy}\f$ normal stress component
  value_type Rxy; ///< Memory variable for \f$R_{xy}\f$ shear stress component
  value_type Rxz; ///< Memory variable for \f$R_{xz}\f$ shear stress component
  value_type Ryz; ///< Memory variable for \f$R_{yz}\f$ shear stress component
  value_type Rkappa; ///< Memory variable for bulk modulus component,
                     ///< \f$R_{zz}\f$ computed from \f$R_{\kappa}\f$,
                     ///< \f$R_{xx}\f$, and \f$R_{yy}\f$
  ///@}

  /**
   * @name Constructors
   *
   */
  ///@{
  /**
   * @brief Default constructor
   *
   * Initializes all memory variable components to zero using the init() method.
   * Suitable for creating empty 3D memory variable objects for later
   * initialization.
   */
  KOKKOS_FUNCTION
  memory_variable() {
    this->init();
    return;
  }

  /**
   * @brief Component-wise constructor for 3D memory variables
   *
   * Directly initializes all six independent 3D stress tensor components
   * with specified values.
   *
   * @param Rxx Memory variable for \f$R_{xx}\f$ component
   * @param Ryy Memory variable for \f$R_{yy}\f$ component
   * @param Rxy Memory variable for \f$R_{xy}\f$ component
   * @param Rxz Memory variable for \f$R_{xz}\f$ component
   * @param Ryz Memory variable for \f$R_{yz}\f$ component
   * @param Rkappa Memory variable for \f$R_{\kappa}\f$ bulk modulus component
   */
  KOKKOS_FUNCTION
  memory_variable(const value_type &Rxx, const value_type &Ryy,
                  const value_type &Rxy, const value_type &Rxz,
                  const value_type &Ryz, const value_type &Rkappa)
      : Rxx(Rxx), Ryy(Ryy), Rxy(Rxy), Rxz(Rxz), Ryz(Ryz), Rkappa(Rkappa) {}

  /**
   * @brief Uniform initialization constructor for 3D
   *
   * Initializes all six memory variable components to the same value.
   * Convenient for testing or uniform initial conditions.
   *
   * @param constant Value to initialize all memory variable components
   */
  KOKKOS_FUNCTION
  memory_variable(const value_type constant)
      : Rxx(constant), Ryy(constant), Rxy(constant), Rxz(constant),
        Ryz(constant), Rkappa(constant) {}

  KOKKOS_FUNCTION
  void init() {
    this->Rxx = value_type(typename value_type::value_type(0));
    this->Ryy = value_type(typename value_type::value_type(0));
    this->Rxy = value_type(typename value_type::value_type(0));
    this->Rxz = value_type(typename value_type::value_type(0));
    this->Ryz = value_type(typename value_type::value_type(0));
    this->Rkappa = value_type(typename value_type::value_type(0));
    return;
  }

  // operator+
  KOKKOS_FUNCTION memory_variable operator+(const memory_variable &rhs) const {
    memory_variable result;
    constexpr int N = specfem::constants::N_SLS;
    for (int i = 0; i < N; ++i) {
      result.Rxx(i) = this->Rxx(i) + rhs.Rxx(i);
      result.Ryy(i) = this->Ryy(i) + rhs.Ryy(i);
      result.Rxy(i) = this->Rxy(i) + rhs.Rxy(i);
      result.Rxz(i) = this->Rxz(i) + rhs.Rxz(i);
      result.Ryz(i) = this->Ryz(i) + rhs.Ryz(i);
      result.Rkappa(i) = this->Rkappa(i) + rhs.Rkappa(i);
    }
    return result;
  }

  // operator+=
  KOKKOS_FUNCTION memory_variable &operator+=(const memory_variable &rhs) {
    constexpr int N = specfem::constants::N_SLS;
    for (int i = 0; i < N; ++i) {
      this->Rxx(i) += rhs.Rxx(i);
      this->Ryy(i) += rhs.Ryy(i);
      this->Rxy(i) += rhs.Rxy(i);
      this->Rxz(i) += rhs.Rxz(i);
      this->Ryz(i) += rhs.Ryz(i);
      this->Rkappa(i) += rhs.Rkappa(i);
    }
    return *this;
  }

  // operator*
  KOKKOS_FUNCTION memory_variable operator*(const type_real &rhs) const {
    memory_variable result;
    constexpr int N = specfem::constants::N_SLS;
    for (int i = 0; i < N; ++i) {
      result.Rxx(i) = this->Rxx(i) * rhs;
      result.Ryy(i) = this->Ryy(i) * rhs;
      result.Rxy(i) = this->Rxy(i) * rhs;
      result.Rxz(i) = this->Rxz(i) * rhs;
      result.Ryz(i) = this->Ryz(i) * rhs;
      result.Rkappa(i) = this->Rkappa(i) * rhs;
    }
    return result;
  }

  // operator==
  KOKKOS_FUNCTION bool operator==(const memory_variable &rhs) const {
    return (this->Rxx == rhs.Rxx) && (this->Ryy == rhs.Ryy) &&
           (this->Rxy == rhs.Rxy) && (this->Rxz == rhs.Rxz) &&
           (this->Ryz == rhs.Ryz) && (this->Rkappa == rhs.Rkappa);
  }
};

} // namespace specfem::point
