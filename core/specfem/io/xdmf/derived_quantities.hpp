#pragma once

#include "specfem/enums.hpp"
#include "specfem/io/xdmf/description.hpp"
#include <string>
#include <vector>

namespace specfem {
namespace io {
namespace xdmf_impl {

/**
 * @brief P and S velocities of an isotropic elastic or Cosserat medium
 *        stored as (kappa, mu, rho).
 *
 * @return `vp` and `vs`
 */
inline std::vector<specfem::io::xdmf::DerivedDescription>
isotropic_elastic_velocities() {
  return {
    { "vp", "sqrt(($0 + 4.0 * $1 / 3.0) / $2)", { "kappa", "mu", "rho" } },
    { "vs", "sqrt($0 / $1)", { "mu", "rho" } }
  };
}

} // namespace xdmf_impl

namespace xdmf {

/**
 * @brief Quantities the XDMF reader derives from the stored properties of one
 *        (medium, property) combination.
 *
 * Each specialization transcribes the accessors of the matching
 * `specfem::medium_container::properties::point_container`
 * (`core/specfem/medium/.../properties.hpp`) into XDMF `Function`
 * expressions over the stored property datasets. The primary template is
 * undefined, so the property writer fails to compile for a combination
 * without a specialization.
 *
 * Xdmf2 evaluates `scalar - array` and `scalar / array` as `array - scalar`
 * and `array / scalar`, so a numeric literal never appears left of `-` or
 * `/` (e.g. \f$ 1 / x \f$ is written `$0 / ($0 * $0)`).
 *
 * @tparam MediumTag Medium of the element group
 * @tparam PropertyTag Property of the element group
 */
template <specfem::element::medium_tag MediumTag,
          specfem::element::property_tag PropertyTag>
struct DerivedQuantities;

/**
 * @brief 2D P-SV isotropic elastic: `vp`, `vs`.
 */
template <>
struct DerivedQuantities<specfem::element::medium_tag::elastic_psv,
                         specfem::element::property_tag::isotropic> {
  /**
   * @brief Derived quantities of the combination.
   * @return One description per derived quantity
   */
  static std::vector<specfem::io::xdmf::DerivedDescription> list() {
    return specfem::io::xdmf_impl::isotropic_elastic_velocities();
  }
};

/**
 * @brief 2D SH isotropic elastic: `vp`, `vs`.
 */
template <>
struct DerivedQuantities<specfem::element::medium_tag::elastic_sh,
                         specfem::element::property_tag::isotropic> {
  /**
   * @brief Derived quantities of the combination.
   * @return One description per derived quantity
   */
  static std::vector<specfem::io::xdmf::DerivedDescription> list() {
    return specfem::io::xdmf_impl::isotropic_elastic_velocities();
  }
};

/**
 * @brief 3D isotropic elastic: `vp`, `vs`.
 */
template <>
struct DerivedQuantities<specfem::element::medium_tag::elastic,
                         specfem::element::property_tag::isotropic> {
  /**
   * @brief Derived quantities of the combination.
   * @return One description per derived quantity
   */
  static std::vector<specfem::io::xdmf::DerivedDescription> list() {
    return specfem::io::xdmf_impl::isotropic_elastic_velocities();
  }
};

/**
 * @brief 2D Cosserat (P-SV-T) isotropic: `vp`, `vs`.
 */
template <>
struct DerivedQuantities<specfem::element::medium_tag::elastic_psv_t,
                         specfem::element::property_tag::isotropic_cosserat> {
  /**
   * @brief Derived quantities of the combination.
   * @return One description per derived quantity
   */
  static std::vector<specfem::io::xdmf::DerivedDescription> list() {
    return specfem::io::xdmf_impl::isotropic_elastic_velocities();
  }
};

/**
 * @brief 3D Cosserat (spin) isotropic: `vp`, `vs`.
 */
template <>
struct DerivedQuantities<specfem::element::medium_tag::elastic_spin,
                         specfem::element::property_tag::isotropic_cosserat> {
  /**
   * @brief Derived quantities of the combination.
   * @return One description per derived quantity
   */
  static std::vector<specfem::io::xdmf::DerivedDescription> list() {
    return specfem::io::xdmf_impl::isotropic_elastic_velocities();
  }
};

/**
 * @brief 2D P-SV anisotropic elastic: vertical `vp`, `vs` from c33, c55.
 */
template <>
struct DerivedQuantities<specfem::element::medium_tag::elastic_psv,
                         specfem::element::property_tag::anisotropic> {
  /**
   * @brief Derived quantities of the combination.
   * @return One description per derived quantity
   */
  static std::vector<specfem::io::xdmf::DerivedDescription> list() {
    return { { "vp", "sqrt($0 / $1)", { "c33", "rho" } },
             { "vs", "sqrt($0 / $1)", { "c55", "rho" } } };
  }
};

/**
 * @brief 2D SH anisotropic elastic: vertical `vp`, `vs` from c33, c55.
 */
template <>
struct DerivedQuantities<specfem::element::medium_tag::elastic_sh,
                         specfem::element::property_tag::anisotropic> {
  /**
   * @brief Derived quantities of the combination.
   * @return One description per derived quantity
   */
  static std::vector<specfem::io::xdmf::DerivedDescription> list() {
    return { { "vp", "sqrt($0 / $1)", { "c33", "rho" } },
             { "vs", "sqrt($0 / $1)", { "c55", "rho" } } };
  }
};

/**
 * @brief 3D anisotropic elastic: `vp`, `vs` from the Voigt-averaged bulk and
 *        shear moduli.
 */
template <>
struct DerivedQuantities<specfem::element::medium_tag::elastic,
                         specfem::element::property_tag::anisotropic> {
  /**
   * @brief Derived quantities of the combination.
   * @return One description per derived quantity
   */
  static std::vector<specfem::io::xdmf::DerivedDescription> list() {
    const std::vector<std::string> operands = { "c11", "c22", "c33", "c12",
                                                "c13", "c23", "c44", "c55",
                                                "c66", "rho" };
    const std::string bulk_modulus =
        "(($0 + $1 + $2 + 2.0 * ($3 + $4 + $5)) / 9.0)";
    const std::string shear_modulus =
        "(($0 + $1 + $2 - $3 - $4 - $5 + 3.0 * ($6 + $7 + $8)) / 15.0)";
    return { { "vp",
               "sqrt((" + bulk_modulus + " + 4.0 * " + shear_modulus +
                   " / 3.0) / $9)",
               operands },
             { "vs", "sqrt(" + shear_modulus + " / $9)", operands } };
  }
};

/**
 * @brief Acoustic isotropic (2D and 3D): `vp`, `rho`.
 */
template <>
struct DerivedQuantities<specfem::element::medium_tag::acoustic,
                         specfem::element::property_tag::isotropic> {
  /**
   * @brief Derived quantities of the combination.
   * @return One description per derived quantity
   */
  static std::vector<specfem::io::xdmf::DerivedDescription> list() {
    return { { "vp", "sqrt($0 * $1)", { "kappa", "rho_inverse" } },
             { "rho", "$0 / ($0 * $0)", { "rho_inverse" } } };
  }
};

/**
 * @brief 2D poroelastic isotropic: `vs` and the average density `rho`.
 *
 * The fast and slow P velocities (`vpI()`, `vpII()`) are plain arithmetic
 * too, but their expressions are longer than the Xdmf2 parser evaluates, so
 * they are not listed.
 */
template <>
struct DerivedQuantities<specfem::element::medium_tag::poroelastic,
                         specfem::element::property_tag::isotropic> {
  /**
   * @brief Derived quantities of the combination.
   * @return One description per derived quantity
   */
  static std::vector<specfem::io::xdmf::DerivedDescription> list() {
    // rho_bar - phi / tortuosity * rho_f
    const std::string a_factor = "($2 - $0 * $2 + $0 * $3 - $0 / $1 * $3)";
    return { { "vs",
               "sqrt($4 / " + a_factor + ")",
               { "phi", "tortuosity", "rho_s", "rho_f", "mu_G" } },
             { "rho", "$1 - $0 * $1 + $0 * $2", { "phi", "rho_s", "rho_f" } } };
  }
};

} // namespace xdmf
} // namespace io
} // namespace specfem
