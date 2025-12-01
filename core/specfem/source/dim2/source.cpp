#include "specfem/source.hpp"
#include "kokkos_abstractions.h"

#include "specfem_setup.hpp"
#include <cmath>

template <>
void specfem::sources::source<specfem::dimension::type::dim2>::set_medium_tag(
    specfem::element::medium_tag medium_tag) {

  auto supported_media_list = this->get_supported_media();
  for (auto &supported_medium : supported_media_list) {
    if (supported_medium == medium_tag) {
      this->medium_tag = medium_tag;
      return;
    }
  }

  const auto gcoord = this->get_global_coordinates();
  const auto lcoord = this->get_local_coordinates();

  std::ostringstream message;

  message << "The element that a " << this->name
          << " is supposed to be placed in \n"
          << "belongs to a medium that is not supported by the " << this->name
          << ".\n"
          << "  Requested medium: " << specfem::element::to_string(medium_tag)
          << "\n"
          << "  Global:\n"
          << "     (x,z)      = " << "(" << gcoord.x << "," << gcoord.z << ")\n"
          << "  Local:\n"
          << "     ispec      = " << lcoord.ispec << "\n"
          << "     (xi,gamma) = " << "(" << lcoord.xi << "," << lcoord.gamma
          << ")\n"
          << "Supported media:\n";
  for (auto &supported_medium : supported_media_list) {
    message << "  - " << specfem::element::to_string(supported_medium) << "\n";
  }
  throw std::runtime_error(message.str());
}

template specfem::sources::source<specfem::dimension::type::dim2>::source(
    YAML::Node &Node, const int nsteps, const type_real dt);
