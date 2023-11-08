#include "constants.hpp"
#include "fortranio/interface.hpp"
#include "material/interface.hpp"
#include "utilities/interface.hpp"
#include <ostream>
#include <tuple>

specfem::material::elastic_material::elastic_material()
    : density(0.0), cs(0.0), cp(0.0), Qkappa(9999.0), Qmu(9999.0),
      compaction_grad(0.0), lambdaplus2mu(0.0), mu(0.0), lambda(0.0),
      kappa(0.0), young(0.0), poisson(0.0){};

std::string specfem::material::elastic_material::print() const {
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

specfem::material::elastic_material::elastic_material(
    const type_real &density, const type_real &cs, const type_real &cp,
    const type_real &Qkappa, const type_real &Qmu,
    const type_real &compaction_grad)
    : density(density), cs(cs), cp(cp), Qkappa(Qkappa), Qmu(Qmu),
      compaction_grad(compaction_grad) {
  this->ispec_type = specfem::enums::element::type::elastic;

  if (this->Qkappa <= 0.0 || this->Qmu <= 0.0) {
    std::runtime_error(
        "negative or null values of Q attenuation factor not allowed; set "
        "them equal to 9999 to indicate no attenuation");
  }

  // Lame parameters
  this->lambdaplus2mu = density * cp * cp;
  this->mu = density * cs * cs;
  this->lambda = this->lambdaplus2mu - 2.0 * this->mu;
  // Bulk modulus
  this->kappa = this->lambda + this->mu;
  // Youngs modulus
  this->young = 9.0 * this->kappa * this->mu / (3.0 * this->kappa + this->mu);
  // Poisson's ratio
  this->poisson = 0.5 * (cp * cp - 2.0 * cs * cs) / (cp * cp - cs * cs);

  if (this->poisson < -1.0 || this->poisson > 0.5)
    std::runtime_error("Poisson's ratio out of range");

  return;
}

specfem::utilities::return_holder
specfem::material::elastic_material::get_properties() const {
  utilities::return_holder holder;
  holder.rho = this->density;
  holder.mu = this->mu;
  holder.kappa = this->kappa;
  holder.qmu = this->Qmu;
  holder.qkappa = this->Qkappa;
  holder.lambdaplus2mu = this->lambdaplus2mu;

  return holder;
}
