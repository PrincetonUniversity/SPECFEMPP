#include "specfem/enums.hpp"
#include "specfem/receivers.hpp"
#include "specfem/setup.hpp"
#include "specfem/utilities.hpp"

std::string
specfem::receivers::receiver<specfem::element::dimension_tag::dim3>::print()
    const {
  std::ostringstream message;
  message << " - Receiver:\n"
          << "      Station Name = " << this->station_name << "\n"
          << "      Network Name = " << this->network_name << "\n"
          << "      Receiver Location: \n"
          << "        x = " << type_real(this->global_coordinates.x) << "\n"
          << "        y = " << type_real(this->global_coordinates.y) << "\n"
          << "        z = " << type_real(this->global_coordinates.z) << "\n";
#ifdef SPECFEM_ENABLE_MPI
  if (this->islice_ >= 0)
    message << "      MPI Rank: " << this->islice_ << "\n";
#endif

  return message.str();
}

bool specfem::receivers::receiver<specfem::element::dimension_tag::dim3>::
operator==(const receiver &other) const {
  return (this->network_name == other.network_name) &&
         (this->station_name == other.station_name) &&
         specfem::utilities::is_close(this->global_coordinates.x,
                                      other.global_coordinates.x) &&
         specfem::utilities::is_close(this->global_coordinates.y,
                                      other.global_coordinates.y) &&
         specfem::utilities::is_close(this->global_coordinates.z,
                                      other.global_coordinates.z);
}
