#ifndef _MATERIAL_PROPERTIES_HPP
#define _MATERIAL_PROPERTIES_HPP

#include "enumerations/specfem_enums.hpp"
#include "specfem_setup.hpp"
#include <exception>
#include <ostream>

namespace specfem {
namespace material {
template <specfem::element::medium_tag type,
          specfem::element::property_tag property>
class properties {
  // disable generic properties struct
  static_assert(type != type, "Invalid material type");
};

template <>
class properties<specfem::element::medium_tag::elastic,
                 specfem::element::property_tag::isotropic> {
public:
  constexpr static auto dimension = specfem::dimension::type::dim2;
  constexpr static auto type = specfem::element::medium_tag::elastic;
  constexpr static auto property = specfem::element::property_tag::isotropic;

  properties(const type_real &density, const type_real &cs, const type_real &cp,
             const type_real &Qkappa, const type_real &Qmu,
             const type_real &compaction_grad)
      : density(density), cs(cs), cp(cp), Qkappa(Qkappa), Qmu(Qmu),
        compaction_grad(compaction_grad), lambdaplus2mu(density * cp * cp),
        mu(density * cs * cs), lambda(lambdaplus2mu - 2.0 * mu),
        kappa(lambda + mu), young(9.0 * kappa * mu / (3.0 * kappa + mu)),
        poisson(0.5 * (cp * cp - 2.0 * cs * cs) / (cp * cp - cs * cs)) {
    if (this->Qkappa <= 0.0 || this->Qmu <= 0.0) {
      std::runtime_error(
          "negative or null values of Q attenuation factor not allowed; set "
          "them equal to 9999 to indicate no attenuation");
    }

    if (this->poisson < -1.0 || this->poisson > 0.5)
      std::runtime_error("Poisson's ratio out of range");
  };

  properties() = default;

  specfem::point::properties<dimension, type, property> get_properties() const {
    return { this->lambdaplus2mu, this->mu, this->density };
  }

  std::string print() const {
    std::ostringstream message;

    message << "- Elastic Material : \n"
            << "    Properties:\n"
            << "      density : " << this->density << "\n"
            << "      cs : " << this->cs << "\n"
            << "      cp : " << this->cp << "\n"
            << "      kappa : " << this->kappa << "\n"
            << "      mu : " << this->mu << "\n"
            << "      Qkappa : " << this->Qkappa << "\n"
            << "      Qmu : " << this->Qmu << "\n"
            << "      lambda : " << this->lambda << "\n"
            << "      mu : " << this->mu << "\n"
            << "      youngs modulus : " << this->young << "\n"
            << "      poisson ratio : " << this->poisson << "\n";

    return message.str();
  }

protected:
  type_real density;
  type_real cs;
  type_real cp;
  type_real Qkappa;
  type_real Qmu;
  type_real compaction_grad;
  type_real lambdaplus2mu;
  type_real mu;
  type_real lambda;
  type_real kappa;
  type_real young;
  type_real poisson;
};

template <>
class properties<specfem::element::medium_tag::acoustic,
                 specfem::element::property_tag::isotropic> {

public:
  constexpr static auto dimension = specfem::dimension::type::dim2;
  constexpr static auto type = specfem::element::medium_tag::acoustic;
  constexpr static auto property = specfem::element::property_tag::isotropic;

  properties(const type_real &density, const type_real &cp,
             const type_real &Qkappa, const type_real &Qmu,
             const type_real &compaction_grad)
      : density(density), cp(cp), Qkappa(Qkappa), Qmu(Qmu),
        compaction_grad(compaction_grad), lambdaplus2mu(density * cp * cp),
        lambda(lambdaplus2mu), kappa(lambda) {
    if (this->Qkappa <= 0.0 || this->Qmu <= 0.0) {
      std::runtime_error(
          "negative or null values of Q attenuation factor not allowed; set "
          "them equal to 9999 to indicate no attenuation");
    }
  }

  properties() = default;

  specfem::point::properties<dimension, type, property> get_properties() const {
    return { 1.0f / static_cast<type_real>(lambdaplus2mu),
             1.0f / static_cast<type_real>(density), this->kappa };
  }

  std::string print() const {
    std::ostringstream message;

    message << "- Acoustic Material : \n"
            << "    Properties:\n"
            << "      density : " << this->density << "\n"
            << "      cp : " << this->cp << "\n"
            << "      kappa : " << this->kappa << "\n"
            << "      Qkappa : " << this->Qkappa << "\n"
            << "      lambda : " << this->lambda << "\n"
            << "      youngs modulus : 0.0 \n"
            << "      poisson ratio :  0.5 \n";

    return message.str();
  }

protected:
  type_real density;
  type_real cp;
  type_real Qkappa;
  type_real Qmu;
  type_real compaction_grad;
  type_real lambdaplus2mu;
  type_real lambda;
  type_real kappa;
};

} // namespace material
} // namespace specfem

#endif
