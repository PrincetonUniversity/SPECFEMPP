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
