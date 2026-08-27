#pragma once

#include "specfem/enums.hpp"
#include "specfem/setup.hpp"
#include "specfem/tags.hpp"
#include <cmath>
#include <sstream>
#include <string>

namespace specfem::medium_container {

/**
 * @defgroup specfem_medium_material_dim3_elastic_anisotropic 3D Elastic
 * Anisotropic Material
 *
 */

/**
 * @ingroup specfem_medium_material_dim3_elastic_anisotropic
 * @brief Material definition for a 3D anisotropic elastic solid.
 *
 * Stores density and the 21 independent stiffnesses of the symmetric
 * \f$6\times6\f$ Voigt matrix. Attenuating anisotropic constitutive physics is
 * intentionally outside this data-container implementation.
 *
 * @tparam MediumTag Physical medium type; must be elastic.
 * @tparam AttenuationTag Attenuation model; currently must be none.
 */
template <specfem::element::medium_tag MediumTag,
          specfem::element::attenuation_tag AttenuationTag>
struct material<
    specfem::element::dimension_tag::dim3, MediumTag,
    specfem::element::property_tag::anisotropic, AttenuationTag,
    std::enable_if_t<specfem::element::is_elastic<MediumTag>::value>>
    : impl::AttenuationValues<specfem::element::dimension_tag::dim3, MediumTag,
                              AttenuationTag> {
  static_assert(AttenuationTag == specfem::element::attenuation_tag::none,
                "3D elastic anisotropic attenuation is not implemented");

public:
  constexpr static auto dimension_tag = specfem::element::dimension_tag::dim3;
  constexpr static auto medium_tag = MediumTag;
  constexpr static auto property_tag =
      specfem::element::property_tag::anisotropic;
  constexpr static auto attenuation_tag = AttenuationTag;

  using attenuation =
      impl::AttenuationValues<dimension_tag, medium_tag, attenuation_tag>;

  /**
   * @brief Construct a 3D anisotropic elastic material.
   * @param density Material density.
   * @param c11 Stiffness \f$c_{11}\f$.
   * @param c12 Stiffness \f$c_{12}\f$.
   * @param c13 Stiffness \f$c_{13}\f$.
   * @param c14 Stiffness \f$c_{14}\f$.
   * @param c15 Stiffness \f$c_{15}\f$.
   * @param c16 Stiffness \f$c_{16}\f$.
   * @param c22 Stiffness \f$c_{22}\f$.
   * @param c23 Stiffness \f$c_{23}\f$.
   * @param c24 Stiffness \f$c_{24}\f$.
   * @param c25 Stiffness \f$c_{25}\f$.
   * @param c26 Stiffness \f$c_{26}\f$.
   * @param c33 Stiffness \f$c_{33}\f$.
   * @param c34 Stiffness \f$c_{34}\f$.
   * @param c35 Stiffness \f$c_{35}\f$.
   * @param c36 Stiffness \f$c_{36}\f$.
   * @param c44 Stiffness \f$c_{44}\f$.
   * @param c45 Stiffness \f$c_{45}\f$.
   * @param c46 Stiffness \f$c_{46}\f$.
   * @param c55 Stiffness \f$c_{55}\f$.
   * @param c56 Stiffness \f$c_{56}\f$.
   * @param c66 Stiffness \f$c_{66}\f$.
   */
  material(const type_real &density, const type_real &c11, const type_real &c12,
           const type_real &c13, const type_real &c14, const type_real &c15,
           const type_real &c16, const type_real &c22, const type_real &c23,
           const type_real &c24, const type_real &c25, const type_real &c26,
           const type_real &c33, const type_real &c34, const type_real &c35,
           const type_real &c36, const type_real &c44, const type_real &c45,
           const type_real &c46, const type_real &c55, const type_real &c56,
           const type_real &c66)
      : density(density), c11(c11), c12(c12), c13(c13), c14(c14), c15(c15),
        c16(c16), c22(c22), c23(c23), c24(c24), c25(c25), c26(c26), c33(c33),
        c34(c34), c35(c35), c36(c36), c44(c44), c45(c45), c46(c46), c55(c55),
        c56(c56), c66(c66) {}

  /** @brief Construct an empty material. */
  material() = default;

  /**
   * @brief Compare material parameters.
   * @param other Material to compare against.
   * @return True when all stored values agree within tolerance.
   */
  bool operator==(const material &other) const {
    constexpr type_real tolerance = static_cast<type_real>(1e-6);
    return std::abs(density - other.density) < tolerance &&
           std::abs(c11 - other.c11) < tolerance &&
           std::abs(c12 - other.c12) < tolerance &&
           std::abs(c13 - other.c13) < tolerance &&
           std::abs(c14 - other.c14) < tolerance &&
           std::abs(c15 - other.c15) < tolerance &&
           std::abs(c16 - other.c16) < tolerance &&
           std::abs(c22 - other.c22) < tolerance &&
           std::abs(c23 - other.c23) < tolerance &&
           std::abs(c24 - other.c24) < tolerance &&
           std::abs(c25 - other.c25) < tolerance &&
           std::abs(c26 - other.c26) < tolerance &&
           std::abs(c33 - other.c33) < tolerance &&
           std::abs(c34 - other.c34) < tolerance &&
           std::abs(c35 - other.c35) < tolerance &&
           std::abs(c36 - other.c36) < tolerance &&
           std::abs(c44 - other.c44) < tolerance &&
           std::abs(c45 - other.c45) < tolerance &&
           std::abs(c46 - other.c46) < tolerance &&
           std::abs(c55 - other.c55) < tolerance &&
           std::abs(c56 - other.c56) < tolerance &&
           std::abs(c66 - other.c66) < tolerance &&
           attenuation::operator==(other);
  }

  /**
   * @brief Compare material parameters for inequality.
   * @param other Material to compare against.
   * @return True when any stored value differs.
   */
  bool operator!=(const material &other) const { return !(*this == other); }

  /**
   * @brief Convert the material to point properties.
   * @return Point properties in the canonical 21-stiffness order, then density.
   */
  specfem::point::properties<
      specfem::tags::Tags<dimension_tag, medium_tag, property_tag, false>>
  get_properties() const {
    return { c11, c12, c13, c14, c15, c16, c22, c23, c24, c25, c26,
             c33, c34, c35, c36, c44, c45, c46, c55, c56, c66, density };
  }

  /**
   * @brief Format the material parameters.
   * @return Human-readable material description.
   */
  std::string print() const {
    std::ostringstream message;
    message << "- 3D Elastic Anisotropic Material :\n"
            << "    Properties:\n"
            << "      density : " << density << "\n"
            << "      c11 : " << c11 << "\n"
            << "      c12 : " << c12 << "\n"
            << "      c13 : " << c13 << "\n"
            << "      c14 : " << c14 << "\n"
            << "      c15 : " << c15 << "\n"
            << "      c16 : " << c16 << "\n"
            << "      c22 : " << c22 << "\n"
            << "      c23 : " << c23 << "\n"
            << "      c24 : " << c24 << "\n"
            << "      c25 : " << c25 << "\n"
            << "      c26 : " << c26 << "\n"
            << "      c33 : " << c33 << "\n"
            << "      c34 : " << c34 << "\n"
            << "      c35 : " << c35 << "\n"
            << "      c36 : " << c36 << "\n"
            << "      c44 : " << c44 << "\n"
            << "      c45 : " << c45 << "\n"
            << "      c46 : " << c46 << "\n"
            << "      c55 : " << c55 << "\n"
            << "      c56 : " << c56 << "\n"
            << "      c66 : " << c66 << "\n"
            << static_cast<const attenuation &>(*this).print();
    return message.str();
  }

protected:
  type_real density; ///< Material density.
  type_real c11;     ///< Stiffness \f$c_{11}\f$.
  type_real c12;     ///< Stiffness \f$c_{12}\f$.
  type_real c13;     ///< Stiffness \f$c_{13}\f$.
  type_real c14;     ///< Stiffness \f$c_{14}\f$.
  type_real c15;     ///< Stiffness \f$c_{15}\f$.
  type_real c16;     ///< Stiffness \f$c_{16}\f$.
  type_real c22;     ///< Stiffness \f$c_{22}\f$.
  type_real c23;     ///< Stiffness \f$c_{23}\f$.
  type_real c24;     ///< Stiffness \f$c_{24}\f$.
  type_real c25;     ///< Stiffness \f$c_{25}\f$.
  type_real c26;     ///< Stiffness \f$c_{26}\f$.
  type_real c33;     ///< Stiffness \f$c_{33}\f$.
  type_real c34;     ///< Stiffness \f$c_{34}\f$.
  type_real c35;     ///< Stiffness \f$c_{35}\f$.
  type_real c36;     ///< Stiffness \f$c_{36}\f$.
  type_real c44;     ///< Stiffness \f$c_{44}\f$.
  type_real c45;     ///< Stiffness \f$c_{45}\f$.
  type_real c46;     ///< Stiffness \f$c_{46}\f$.
  type_real c55;     ///< Stiffness \f$c_{55}\f$.
  type_real c56;     ///< Stiffness \f$c_{56}\f$.
  type_real c66;     ///< Stiffness \f$c_{66}\f$.
};

} // namespace specfem::medium_container
