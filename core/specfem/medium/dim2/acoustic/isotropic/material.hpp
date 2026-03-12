#pragma once

#include "specfem/element.hpp"
#include "specfem/setup.hpp"
#include <exception>
#include <iostream>
#include <ostream>
#include <sstream>

namespace specfem {
namespace medium_container {

namespace impl {

template <specfem::element::dimension_tag DimensionTag>
struct AttenuationValues<
    DimensionTag, specfem::element::medium_tag::acoustic,
    specfem::element::attenuation_tag::constant_isotropic> {
public:
  type_real Qkappa; ///< Attenuation factor for bulk modulus

  AttenuationValues() = default;

  AttenuationValues(const type_real &Qkappa) : Qkappa(Qkappa) {
    if (this->Qkappa <= 0.0) {
      throw std::runtime_error(
          "negative or null values of Q attenuation factor not allowed; set "
          "them equal to 9999 to indicate no attenuation");
    }
  };

  bool operator==(const AttenuationValues &other) const {
    return (std::abs(this->Qkappa - other.Qkappa) < 1e-6);
  }

  std::string print() const {
    std::ostringstream message;

    message << "      Qkappa : " << this->Qkappa << "\n";

    return message.str();
  }
};

} // namespace impl

/**
 * @brief Template specialization for acoustic isotropic material properties
 *
 */
template <specfem::element::dimension_tag DimensionTag,
          specfem::element::attenuation_tag AttenuationTag>
class material<DimensionTag, specfem::element::medium_tag::acoustic,
               specfem::element::property_tag::isotropic, AttenuationTag>
    : impl::AttenuationValues<DimensionTag,
                              specfem::element::medium_tag::acoustic,
                              AttenuationTag> {

public:
  constexpr static auto dimension_tag =
      DimensionTag; ///< Dimension of the material
  constexpr static auto medium_tag =
      specfem::element::medium_tag::acoustic; ///< Medium tag
  constexpr static auto property_tag =
      specfem::element::property_tag::isotropic;          ///< Property tag
  constexpr static auto attenuation_tag = AttenuationTag; ///< Attenuation tag

  using attenuation = impl::AttenuationValues<
      DimensionTag, specfem::element::medium_tag::acoustic, AttenuationTag>;

  /**
   * @name Constructors
   */
  ///@{
  /**
   * @brief Construct a new acoustic isotropic material
   *
   * @param density Density of the material
   * @param cp Compressional wave speed
   * @param Qkappa Attenuation factor for bulk modulus
   * @param Qmu Attenuation factor for shear modulus
   * @param compaction_grad Compaction gradient
   */
  template <
      specfem::element::attenuation_tag T = AttenuationTag,
      std::enable_if_t<T == specfem::element::attenuation_tag::none, int> = 0>
  material(const type_real &density, const type_real &cp,
           const type_real &compaction_grad)
      : density(density), cp(cp), compaction_grad(compaction_grad) {
    this->kappa = density * cp * cp;
  }

  template <
      specfem::element::attenuation_tag T = AttenuationTag,
      std::enable_if_t<
          T == specfem::element::attenuation_tag::constant_isotropic, int> = 0>
  material(const type_real &density, const type_real &cp,
           const type_real &Qkappa, const type_real &compaction_grad)
      : density(density), cp(cp), compaction_grad(compaction_grad),
        attenuation(Qkappa) {
    this->kappa = density * cp * cp;
  }

  /**
   * @brief Check if 2 materials have the same properties
   *
   * @param other Material to compare with
   * @return true If the materials have the same properties
   */
  bool operator==(
      const material<dimension_tag, specfem::element::medium_tag::acoustic,
                     specfem::element::property_tag::isotropic, attenuation_tag>
          &other) const {

    return (std::abs(this->density - other.density) < 1e-6 &&
            std::abs(this->cp - other.cp) < 1e-6 &&
            attenuation::operator==(other) &&
            std::abs(this->compaction_grad - other.compaction_grad) < 1e-6);
  }

  /**
   * @brief Check if 2 materials have different properties
   *
   * @param other Material to compare with
   * @return true If the materials have different properties
   */
  bool operator!=(
      const material<dimension_tag, specfem::element::medium_tag::acoustic,
                     specfem::element::property_tag::isotropic, attenuation_tag>
          &other) const {
    return !(*this == other);
  }

  /**
   * @brief Default constructor
   *
   */
  material() = default;
  ///@}

  ~material() = default;

  /**
   * @brief Get the properties of the material
   *
   * @return specfem::point::properties Material properties
   */
  template <specfem::element::attenuation_tag T = AttenuationTag,
            typename =
                std::enable_if_t<T == specfem::element::attenuation_tag::none> >
  inline specfem::point::properties<dimension_tag, medium_tag, property_tag,
                                    false>
  get_properties() const {
    return { static_cast<type_real>(1.0) / static_cast<type_real>(density),
             this->kappa };
  }

  /**
   * @brief Print the material properties
   *
   * @return std::string Formatted material properties
   */
  inline std::string print() const {
    std::ostringstream message;

    message << "- Acoustic Material : \n"
            << "    Properties:\n"
            << "      density : " << this->density << "\n"
            << "      cp : " << this->cp << "\n"
            << "      kappa : " << this->kappa << "\n"
            << static_cast<const attenuation &>(*this).print()
            << "      youngs modulus : 0.0 \n"
            << "      poisson ratio :  0.5 \n";

    return message.str();
  }

private:
  type_real density;         ///< Density of the material
  type_real cp;              ///< Compressional wave speed
  type_real compaction_grad; ///< Compaction gradient
  type_real kappa;           ///< Bulk modulus
};

} // namespace medium_container
} // namespace specfem
