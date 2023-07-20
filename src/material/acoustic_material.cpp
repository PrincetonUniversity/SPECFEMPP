#include "constants.hpp"
#include "fortranio/interface.hpp"
#include "material/interface.hpp"
#include "utilities/interface.hpp"
#include <ostream>
#include <tuple>

specfem::material::acoustic_material::acoustic_material()
    : density(0.0), cp(0.0), Qkappa(9999.0), Qmu(9999.0), compaction_grad(0.0),
      lambdaplus2mu(0.0), lambda(0.0), kappa(0.0), young(0.0), poisson(0.0){};

std::string specfem::material::acoustic_material::print() const {
  std::ostringstream message;

  message << "- Acoustic Material : \n"
          << "    Properties:\n"
          << "      density : " << this->density << "\n"
          << "      cp : " << this->cp << "\n"
          << "      kappa : " << this->kappa << "\n"
          << "      Qkappa : " << this->Qkappa << "\n"
          << "      lambda : " << this->lambda << "\n"
          << "      youngs modulus : " << this->young << "\n"
          << "      poisson ratio : " << this->poisson << "\n";

  return message.str();
}

specfem::material::acoustic_material::acoustic_material(
    const type_real &density, const type_real &cp, const type_real &Qkappa,
    const type_real &Qmu, const type_real &compaction_grad)
    : density(density), cp(cp), Qkappa(Qkappa), Qmu(Qmu),
      compaction_grad(compaction_grad) {

  this->ispec_type = specfem::enums::element::acoustic;

  if (this->Qkappa <= 0.0 || this->Qmu <= 0.0) {
    std::runtime_error(
        "negative or null values of Q attenuation factor not allowed; set "
        "them equal to 9999 to indicate no attenuation");
  }

  // Lame parameters: note these are identical since mu = 0
  this->lambdaplus2mu = density * cp * cp;
  this->lambda = this->lambdaplus2mu;

  // Bulk modulus
  this->kappa = this->lambda;
  // Youngs modulus - for fluid always 0
  this->young = 0.0;
  // Poisson's ratio - for fluid always 0
  this->poisson = 0.5;

  if (this->poisson < -1.0 || this->poisson > 0.5)
    std::runtime_error("Poisson's ratio out of range");

  return;
}

specfem::utilities::return_holder
specfem::material::acoustic_material::get_properties() {
  utilities::return_holder holder;
  holder.rho = this->density;
  holder.kappa = this->kappa;
  holder.qmu = this->Qmu;
  holder.qkappa = this->Qkappa;
  holder.lambdaplus2mu = this->lambdaplus2mu;

  return holder;
}
