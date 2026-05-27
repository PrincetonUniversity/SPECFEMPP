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
          << "      Network Name = " << this->network_name << "\n";
  if (this->read_coordinates_) {
    message << "      Input Coordinates: " << this->read_coordinates_->print()
            << "\n";
  }
  // Only print resolved global coordinates when they have been set
  if (!this->read_coordinates_ || this->islice_ >= 0) {
    message << "      Receiver Location: \n"
            << "        x = " << type_real(this->global_coordinates.x) << "\n"
            << "        y = " << type_real(this->global_coordinates.y) << "\n"
            << "        z = " << type_real(this->global_coordinates.z) << "\n";
  }
#ifdef SPECFEM_ENABLE_MPI
  if (this->islice_ >= 0)
    message << "      MPI Rank: " << this->islice_ << "\n";
#endif

  return message.str();
}

bool specfem::receivers::receiver<specfem::element::dimension_tag::dim3>::
operator==(const receiver &other) const {
  // Compare input coordinates when available
  const auto *c1 = this->get_read_coordinates();
  const auto *c2 = other.get_read_coordinates();
  bool coords_equal = (c1 && c2) ? (*c1 == *c2) : (!c1 && !c2);

  return coords_equal && (this->network_name == other.network_name) &&
         (this->station_name == other.station_name);
}
