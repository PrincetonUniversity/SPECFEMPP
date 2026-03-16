#pragma once

#include "specfem/constants.hpp"
#include "specfem/datatype/hertz.hpp"
#include "specfem/datatype/radians.hpp"
#include "specfem/datatype/seconds.hpp"

namespace specfem::datatype {

inline Omega::operator Seconds() const {
  // convert angular frequency (rad/s) to period (s)
  return Seconds(static_cast<type_real>(2.0 * specfem::constants::pi) /
                 static_cast<type_real>(*this));
};

inline Omega::operator Hertz() const {
  // convert angular frequency (rad/s) to frequency (Hz)
  return Hertz(static_cast<type_real>(*this) /
               static_cast<type_real>(2.0 * specfem::constants::pi));
};

inline Seconds::operator Omega() const {
  // convert period (s) to angular frequency (rad/s)
  return Omega(static_cast<type_real>(2.0 * specfem::constants::pi) /
               static_cast<type_real>(*this));
};

inline Seconds::operator Hertz() const {
  // convert period (s) to frequency (Hz)
  return Hertz(static_cast<type_real>(1.0) / static_cast<type_real>(*this));
};

inline Hertz::operator Omega() const {
  // convert frequency (Hz) to angular frequency (rad/s)
  return Omega(static_cast<type_real>(*this) *
               static_cast<type_real>(2.0 * specfem::constants::pi));
};

inline Hertz::operator Seconds() const {
  // convert frequency (Hz) to period (s)
  return Seconds(static_cast<type_real>(1.0) / static_cast<type_real>(*this));
};
} // namespace specfem::datatype
