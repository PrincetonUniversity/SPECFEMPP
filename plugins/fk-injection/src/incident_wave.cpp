#include "fk_injection/incident_wave.hpp"
#include <cmath>

fk_injection::type_real
fk_injection::IncidentWave::ray_parameter(type_real halfspace_velocity) const {
  return std::sin(take_off_theta) / halfspace_velocity;
}
