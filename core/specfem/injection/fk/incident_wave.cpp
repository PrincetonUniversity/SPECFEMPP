#include "specfem/injection/fk/incident_wave.hpp"
#include <cmath>

type_real specfem::injection::fk::IncidentWave::ray_parameter(
    type_real halfspace_velocity) const {
  return std::sin(take_off_theta) / halfspace_velocity;
}
