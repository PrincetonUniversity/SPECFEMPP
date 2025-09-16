#include "enumerations/interface.hpp"
#include "globals.h"
#include "kokkos_abstractions.h"
#include "specfem/receivers.hpp"
#include "specfem_setup.hpp"
#include "utilities/interface.hpp"

std::string
specfem::receivers::receiver<specfem::dimension::type::dim2>::print() const {
  std::ostringstream message;
  message << " - Receiver:\n"
          << "      Station Name = " << this->station_name << "\n"
          << "      Network Name = " << this->network_name << "\n"
          << "      Receiver Location: \n"
          << "        x = " << type_real(this->x) << "\n"
          << "        z = " << type_real(this->z) << "\n";

  return message.str();
}

bool specfem::receivers::receiver<specfem::dimension::type::dim2>::operator==(
    const receiver &other) const {
  return (this->network_name == other.network_name) &&
         (this->station_name == other.station_name) &&
         specfem::utilities::is_close(this->x, other.x) &&
         specfem::utilities::is_close(this->z, other.z) &&
         specfem::utilities::is_close(this->angle, other.angle);
}
