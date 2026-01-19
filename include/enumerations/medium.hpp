#pragma once

#include "enumerations/dimension.hpp"
#include "specfem/utilities/errors.hpp"
#include <array>
#include <tuple>

namespace specfem {
namespace element {

/**
 * @brief Element medium types for physics simulations.
 *
 * Defines wave propagation physics: elastic (P/SV/SH waves), acoustic (pressure
 * waves), poroelastic (fluid-solid interaction), and electromagnetic (TE/TM
 * modes).
 */
enum class medium_tag {
  elastic_psv,        ///< 2D elastic medium with P and SV waves
  elastic_sh,         ///< 2D elastic medium with SH waves
  elastic_psv_t,      ///< 2D elastic PSV with transverse spin (Cosserat)
  acoustic,           ///< Acoustic medium (pressure waves)
  poroelastic,        ///< Poroelastic medium (Biot theory)
  electromagnetic_te, ///< 2D electromagnetic TE modes
  elastic,            ///< 3D elastic medium (full displacement field)
  elastic_spin,       ///< Elastic medium with spin dynamics
  electromagnetic,    ///< Electromagnetic medium (TE and TM modes)
};

/**
 * @brief Material property symmetries.
 *
 * Controls material tensor structure: isotropic (scalar properties),
 * anisotropic (full tensor), isotropic_cosserat (with microrotation).
 */
enum class property_tag {
  isotropic,         ///< Isotropic material (scalar properties)
  anisotropic,       ///< Anisotropic material (full tensor)
  isotropic_cosserat ///< Isotropic Cosserat material (with microrotation)
};

/**
 * @brief Boundary condition types for domain edges.
 *
 * Defines how waves interact with domain boundaries: free surfaces,
 * absorbing conditions (Stacey), and composite boundary treatments.
 */
enum class boundary_tag {
  // primary boundaries
  none,                  ///< No boundary condition
  acoustic_free_surface, ///< Acoustic free surface (zero pressure)
  stacey,                ///< Stacey absorbing boundary condition

  // composite boundaries
  composite_stacey_dirichlet ///< Combined Stacey-Dirichlet boundary
};

/**
 * @brief Element physics attributes for different media.
 *
 * Template specializations define field components, physics flags,
 * and computational requirements for each medium type.
 *
 * @tparam Dimension Spatial dimension (2D or 3D)
 * @tparam MediumTag Medium physics type
 *
 * @code
 * // Get attributes for 2D elastic PSV medium
 * using attrs = specfem::element::attributes<
 *   specfem::dimension::type::dim2,
 *   specfem::element::medium_tag::elastic_psv>;
 * static_assert(attrs::components == 2); // u_x, u_z components
 * static_assert(attrs::dimension == 2);
 * @endcode
 */
template <specfem::dimension::type Dimension,
          specfem::element::medium_tag MediumTag>
class attributes {
  static_assert(specfem::utilities::always_false<Dimension, MediumTag>,
                "Unregistered attributes tag! Please add a specialization for "
                "dimension/medium enum value.");
};

// ===========================================================================
// @brief 2D attributes specialization
// ===========================================================================
/**
 * @brief Attributes for 2D elastic PSV waves (P and SV components).
 *
 * Handles in-plane wave propagation with displacement components u_x and u_z.
 */
template <>
class attributes<specfem::dimension::type::dim2,
                 specfem::element::medium_tag::elastic_psv> {

public:
  inline static constexpr int dimension = 2;  ///< Spatial dimension
  inline static constexpr int components = 2; ///< Field components (u_x, u_z)

  constexpr static bool has_damping_force = false; ///< No damping physics
  inline constexpr static bool has_cosserat_stress =
      false; ///< No Cosserat stress
  inline constexpr static bool has_cosserat_couple_stress =
      false; ///< No couple stress
};

/**
 * @brief Attributes for 2D elastic SH waves (out-of-plane shear).
 *
 * Handles anti-plane wave propagation with single displacement component u_y.
 */
template <>
class attributes<specfem::dimension::type::dim2,
                 specfem::element::medium_tag::elastic_sh> {

public:
  inline static constexpr int dimension = 2;  ///< Spatial dimension
  inline static constexpr int components = 1; ///< Field components (u_y)

  inline constexpr static bool has_damping_force =
      false; ///< No damping physics
  inline constexpr static bool has_cosserat_stress =
      false; ///< No Cosserat stress
  inline constexpr static bool has_cosserat_couple_stress =
      false; ///< No couple stress
};

/**
 * @brief Attributes for 2D elastic PSV with transverse spin (Cosserat medium).
 *
 * Extends PSV waves with microrotation physics for granular/micropolar
 * materials. Includes displacement (u_x, u_z) and rotation (ω_y) components.
 */
template <>
class attributes<specfem::dimension::type::dim2,
                 specfem::element::medium_tag::elastic_psv_t> {

public:
  inline static constexpr int dimension = 2; ///< Spatial dimension
  inline static constexpr int components =
      3; ///< Field components (u_x, u_z, ω_y)

  inline constexpr static bool has_damping_force =
      false; ///< No damping physics
  inline constexpr static bool has_cosserat_stress =
      true; ///< Has Cosserat stress
  inline constexpr static bool has_cosserat_couple_stress =
      true; ///< Has couple stress
};

/**
 * @brief Attributes for 2D acoustic waves (pressure field).
 *
 * Handles compressional wave propagation in fluid media with scalar pressure.
 */
template <>
class attributes<specfem::dimension::type::dim2,
                 specfem::element::medium_tag::acoustic> {

public:
  inline static constexpr int dimension = 2;  ///< Spatial dimension
  inline static constexpr int components = 1; ///< Field components (pressure)

  inline constexpr static bool has_damping_force =
      false; ///< No damping physics
  inline constexpr static bool has_cosserat_stress =
      false; ///< No Cosserat stress
  inline constexpr static bool has_cosserat_couple_stress =
      false; ///< No couple stress
};

/**
 * @brief Attributes for 2D poroelastic media (Biot theory).
 *
 * Couples solid skeleton and fluid flow with displacement and pressure fields.
 * Components: solid displacement (u_s^x, u_s^z) and relative fluid motion (w^x,
 * w^z).
 */
template <>
class attributes<specfem::dimension::type::dim2,
                 specfem::element::medium_tag::poroelastic> {
public:
  inline static constexpr int dimension = 2; ///< Spatial dimension
  inline static constexpr int components =
      4; ///< Field components (u_s^x, u_s^z, w^x, w^z)

  inline constexpr static bool has_damping_force = true; ///< Has fluid damping
  inline constexpr static bool has_cosserat_stress =
      false; ///< No Cosserat stress
  inline constexpr static bool has_cosserat_couple_stress =
      false; ///< No couple stress
};

/**
 * @brief Attributes for 2D electromagnetic TE waves (transverse electric).
 *
 * Handles electromagnetic wave propagation with electric field components E_x
 * and E_y.
 */
template <>
class attributes<specfem::dimension::type::dim2,
                 specfem::element::medium_tag::electromagnetic_te> {
public:
  inline static constexpr int dimension = 2;  ///< Spatial dimension
  inline static constexpr int components = 2; ///< Field components (E_x, E_y)

  inline constexpr static bool has_damping_force =
      false; ///< No damping physics
  inline constexpr static bool has_cosserat_stress =
      false; ///< No Cosserat stress
  inline constexpr static bool has_cosserat_couple_stress =
      false; ///< No couple stress
};

// ===========================================================================
// @brief 3D attributes specialization
// ===========================================================================

/**
 * @brief Attributes for 3D acoustic waves (pressure field).
 *
 * Handles compressional wave propagation in 3D fluid media with scalar
 * pressure.
 */
template <>
class attributes<specfem::dimension::type::dim3,
                 specfem::element::medium_tag::acoustic> {
public:
  inline static constexpr int dimension = 3;  ///< Spatial dimension
  inline static constexpr int components = 1; ///< Field components (pressure)

  constexpr static bool has_damping_force = false; ///< No damping physics
  inline constexpr static bool has_cosserat_stress =
      false; ///< No Cosserat stress
  inline constexpr static bool has_cosserat_couple_stress =
      false; ///< No couple stress
};

/**
 * @brief Attributes for 3D elastic waves (full displacement field).
 *
 * Handles 3D elastic wave propagation with displacement components u_x, u_y,
 * u_z.
 */
template <>
class attributes<specfem::dimension::type::dim3,
                 specfem::element::medium_tag::elastic> {
public:
  inline static constexpr int dimension = 3; ///< Spatial dimension
  inline static constexpr int components =
      3; ///< Field components (u_x, u_y, u_z)

  constexpr static bool has_damping_force = false; ///< No damping physics
  inline constexpr static bool has_cosserat_stress =
      false; ///< No Cosserat stress
  inline constexpr static bool has_cosserat_couple_stress =
      false; ///< No couple stress
};

/**
 * @brief Convert medium, property, and boundary tags to string.
 *
 * @param medium Medium type
 * @param property_tag Property type
 * @param boundary_tag Boundary condition type
 * @return Combined string representation
 */
const std::string to_string(const medium_tag &medium,
                            const property_tag &property_tag,
                            const boundary_tag &boundary_tag);

/**
 * @brief Convert medium and property tags to string.
 *
 * @param medium Medium type
 * @param property_tag Property type
 * @return Combined string representation
 */
const std::string to_string(const medium_tag &medium,
                            const property_tag &property_tag);

/**
 * @brief Convert medium tag to string.
 *
 * @param medium Medium type
 * @return String representation
 */
const std::string to_string(const medium_tag &medium);

/**
 * @brief Convert property tag to string.
 *
 * @param property Property type
 * @return String representation
 */
const std::string to_string(const property_tag &property);

/**
 * @brief Convert boundary tag to string.
 *
 * @param boundary Boundary condition type
 * @return String representation
 */
const std::string to_string(const boundary_tag &boundary);

/**
 * @brief Type trait to identify elastic media.
 *
 * @tparam MediumTag Medium type to check
 * @return std::true_type if elastic, std::false_type otherwise
 *
 * @code
 * static_assert(is_elastic<medium_tag::elastic_psv>::value);
 * static_assert(!is_elastic<medium_tag::acoustic>::value);
 * @endcode
 */
template <specfem::element::medium_tag MediumTag>
using is_elastic = typename std::conditional_t<
    (MediumTag == specfem::element::medium_tag::elastic ||
     MediumTag == specfem::element::medium_tag::elastic_psv ||
     MediumTag == specfem::element::medium_tag::elastic_sh ||
     MediumTag == specfem::element::medium_tag::elastic_psv_t),
    std::true_type, std::false_type>::type;

/**
 * @brief Type trait to identify electromagnetic media.
 *
 * @tparam MediumTag Medium type to check
 * @return std::true_type if electromagnetic, std::false_type otherwise
 */
template <specfem::element::medium_tag MediumTag>
using is_electromagnetic = typename std::conditional_t<
    (MediumTag == specfem::element::medium_tag::electromagnetic ||
     MediumTag == specfem::element::medium_tag::electromagnetic_te),
    std::true_type, std::false_type>::type;

/**
 * @brief Parse medium tag from string representation.
 *
 * @param medium_tag String representation of medium type
 * @return Corresponding medium_tag enumeration value
 * @throws std::runtime_error if string is not recognized
 */
specfem::element::medium_tag from_string(const std::string &medium_tag);

} // namespace element
} // namespace specfem
