#include "enums.h"
#include "fortranio/interface.hpp"
#include "material/interface.hpp"
#include "utils.h"
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

void specfem::material::elastic_material::assign(
    utilities::input_holder &holder) {
  this->ispec_type = specfem::elements::elastic;
  // density
  this->density = holder.val0;
  // P and S velocity
  this->cp = holder.val1;
  this->cs = holder.val2;
  this->compaction_grad = holder.val3;

  // Qkappa and Qmu values
  this->Qkappa = holder.val5;
  this->Qmu = holder.val6;
  if (this->Qkappa <= 0.0 || this->Qmu <= 0.0) {
    std::runtime_error(
        "negative or null values of Q attenuation factor not allowed; set "
        "them equal to 9999 to indicate no attenuation");
  }
  // Lame parameters
  this->lambdaplus2mu = this->density * this->cp * this->cp;
  this->mu = this->density * this->cs * this->cs;
  this->lambda = this->lambdaplus2mu - 2.0 * this->mu;
  // Bulk modulus
  this->kappa = this->lambda + this->mu;
  // Youngs modulus
  this->young = 9.0 * this->kappa * this->mu / (3.0 * this->kappa + this->mu);
  // Poisson's ratio
  this->poisson = 0.5 * (this->cp * this->cp - 2.0 * this->cs * this->cs) /
                  (this->cp * this->cp - this->cs * this->cs);

  if (this->poisson < -1.0 || this->poisson > 0.5)
    std::runtime_error("Poisson's ratio out of range");
}

specfem::utilities::return_holder
specfem::material::elastic_material::get_properties() {
  utilities::return_holder holder;
  holder.rho = this->density;
  holder.mu = this->mu;
  holder.kappa = this->kappa;
  holder.qmu = this->Qmu;
  holder.qkappa = this->Qkappa;
  holder.lambdaplus2mu = this->lambdaplus2mu;

  return holder;
}
