#include "enumerations/interface.hpp"
#include "globals.h"
#include "kokkos_abstractions.h"
#include "specfem/receivers.hpp"
#include "specfem_setup.hpp"
#include "utilities/interface.hpp"

std::string
specfem::receivers::receiver<specfem::dimension::type::dim3>::print() const {
  std::ostringstream message;
  message << " - Receiver:\n"
          << "      Station Name = " << this->station_name << "\n"
          << "      Network Name = " << this->network_name << "\n"
          << "      Receiver Location: \n"
          << "        x = " << type_real(this->x) << "\n"
          << "        y = " << type_real(this->y) << "\n"
          << "        z = " << type_real(this->z) << "\n";

  return message.str();
}
