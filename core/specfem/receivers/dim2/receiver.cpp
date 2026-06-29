#include "specfem/enums.hpp"
#include "specfem/receivers.hpp"
#include "specfem/setup.hpp"
#include "specfem/utilities.hpp"

std::string
specfem::receivers::receiver<specfem::element::dimension_tag::dim2>::print()
    const {
  std::ostringstream message;
  message << "- " << this->network_name << "." << this->station_name;
  if (this->read_coordinates_) {
    message << ", Coordinates: " << this->read_coordinates_->print();
  }
#ifdef SPECFEM_ENABLE_MPI
  if (this->partition_index_ >= 0)
    message << ", MPI Rank: " << this->partition_index_;
#endif
  message << "\n";
  return message.str();
}

bool specfem::receivers::receiver<specfem::element::dimension_tag::dim2>::
operator==(const receiver &other) const {
  // Compare input coordinates when available
  const auto *c1 = this->get_read_coordinates();
  const auto *c2 = other.get_read_coordinates();
  bool coords_equal = (c1 && c2) ? (*c1 == *c2) : (!c1 && !c2);

  return coords_equal && (this->network_name == other.network_name) &&
         (this->station_name == other.station_name) &&
         specfem::utilities::is_close(this->angle, other.angle);
}
