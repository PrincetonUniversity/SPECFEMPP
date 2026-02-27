#include "specfem/constants.hpp"
#include "specfem/data_access/accessor.hpp"
#include "specfem/datatype.hpp"

namespace specfem::point {

template <specfem::element::dimension_tag DimensionTag,
          specfem::element::medium_tag MediumTag,
          specfem::element::attenuation_tag AttenuationTag, bool UseSIMD>
struct memory_variable;

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
  constexpr static bool store_jacobian = false;
  ///@}

  value_type Rxx;    ///< Memory variable for Rxx component, Rzz compute from
                     ///< Rkappa and Rxx
  value_type Rxz;    ///< Memory variable for Rxz component
  value_type Rkappa; ///< Memory variable for Rkappa component

  /**
   * @name Constructors
   *
   */
  ///@{
  /**
   * @brief Default constructor
   *
   */
  KOKKOS_FUNCTION
  memory_variable() {
    this->init();
    return;
  }

  KOKKOS_FUNCTION
  memory_variable(const value_type &Rxx, const value_type &Rxz,
                  const value_type &Rkappa)
      : Rxx(Rxx), Rxz(Rxz), Rkappa(Rkappa) {}

  /**
   * @brief Constructor with constant value
   *
   * @param constant Value to initialize all members to
   */
  KOKKOS_FUNCTION
  memory_variable(const value_type constant)
      : Rxx(constant), Rxz(constant), Rkappa(constant) {}

  KOKKOS_FUNCTION
  void init() {
    this->Rxx = value_type(typename value_type::value_type(0));
    this->Rxz = value_type(typename value_type::value_type(0));
    this->Rkappa = value_type(typename value_type::value_type(0));
    return;
  }

  // operator+
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

  // operator+=
  KOKKOS_FUNCTION memory_variable &operator+=(const memory_variable &rhs) {
    constexpr int N = specfem::constants::N_SLS;
    for (int i = 0; i < N; ++i) {
      this->Rxx(i) += rhs.Rxx(i);
      this->Rxz(i) += rhs.Rxz(i);
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
      result.Rxz(i) = this->Rxz(i) * rhs;
      result.Rkappa(i) = this->Rkappa(i) * rhs;
    }
    return result;
  }

  // operator==
  KOKKOS_FUNCTION bool operator==(const memory_variable &rhs) const {
    return (this->Rxx == rhs.Rxx) && (this->Rxz == rhs.Rxz) &&
           (this->Rkappa == rhs.Rkappa);
  }
};

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
  constexpr static bool store_jacobian = false;
  ///@}

  value_type Rxx;    ///< Memory variable for Rxx component
  value_type Ryy;    ///< Memory variable for Ryy component, Rzz compute from
                     ///< Rkappa, Rxx and Ryy
  value_type Rxy;    ///< Memory variable for Rxy component
  value_type Rxz;    ///< Memory variable for Rxz component
  value_type Ryz;    ///< Memory variable for Ryz component
  value_type Rkappa; ///< Memory variable for Rkappa component

  /**
   * @name Constructors
   *
   */
  ///@{
  /**
   * @brief Default constructor
   *
   */
  KOKKOS_FUNCTION
  memory_variable() {
    this->init();
    return;
  }

  KOKKOS_FUNCTION
  memory_variable(const value_type &Rxx, const value_type &Ryy,
                  const value_type &Rxy, const value_type &Rxz,
                  const value_type &Ryz, const value_type &Rkappa)
      : Rxx(Rxx), Ryy(Ryy), Rxy(Rxy), Rxz(Rxz), Ryz(Ryz), Rkappa(Rkappa) {}

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
