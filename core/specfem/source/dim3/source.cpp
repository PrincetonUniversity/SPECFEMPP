#include "specfem/source.hpp"
#include "kokkos_abstractions.h"
#include "specfem_mpi/interface.hpp"
#include "specfem_setup.hpp"
#include <cmath>

specfem::sources::source<specfem::dimension::type::dim3>::source(
    YAML::Node &Node, const int nsteps, const type_real dt)
    : global_coordinates(Node["x"].as<type_real>(), Node["y"].as<type_real>(),
                         Node["z"].as<type_real>()) {

  // Read source time function
  if (YAML::Node Dirac = Node["Dirac"]) {
    this->forcing_function = std::make_unique<specfem::forcing_function::Dirac>(
        Dirac, nsteps, dt, false);
  } else if (YAML::Node Ricker = Node["Ricker"]) {
    this->forcing_function =
        std::make_unique<specfem::forcing_function::Ricker>(Ricker, nsteps, dt,
                                                            false);
  } else if (YAML::Node dGaussian = Node["dGaussian"]) {
    this->forcing_function =
        std::make_unique<specfem::forcing_function::dGaussian>(
            dGaussian, nsteps, dt, false);
  } else if (YAML::Node external = Node["External"]) {
    this->forcing_function =
        std::make_unique<specfem::forcing_function::external>(external, nsteps,
                                                              dt);
  } else {
    throw std::runtime_error("Error: source time function not recognized");
  }

  return;
}

void specfem::sources::source<specfem::dimension::type::dim3>::set_medium_tag(
    specfem::element::medium_tag medium_tag) {

  auto supported_media_list = this->get_supported_media();
  for (auto &supported_medium : supported_media_list) {
    if (supported_medium == medium_tag) {
      this->medium_tag = medium_tag;
      return;
    }
  }

  std::ostringstream message;

  const auto gcoord = this->get_global_coordinates();
  const auto lcoord = this->get_local_coordinates();

  message << "The element that a " << this->name
          << " is supposed to be placed in \n"
          << "belongs to a medium that is not supported by the " << this->name
          << ".\n"
          << "  Requested medium: " << specfem::element::to_string(medium_tag)
          << "\n"
          << "  Global:\n"
          << "    (x,y,z)      = " << "(" << gcoord.x << "," << gcoord.y << ","
          << gcoord.z << ")\n"
          << "  Local:\n"
          << "   ispec         = " << lcoord.ispec << "\n"
          << "   (xi,eta,zeta) = " << "(" << lcoord.xi << "," << lcoord.eta
          << "," << lcoord.gamma << ")\n"
          << "Supported media:\n";
  for (auto &supported_medium : supported_media_list) {
    message << "  - " << specfem::element::to_string(supported_medium) << "\n";
  }
  throw std::runtime_error(message.str());
}
