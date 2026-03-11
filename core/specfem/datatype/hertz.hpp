#pragma once

namespace specfem::datatype {

// Forward declarations

class Omega;
class Seconds;

class Hertz : public type_real {
public:
  using type_real::type_real; // Inherit constructors

  // Allow implicit conversion to type_real
  explicit inline operator type_real() const {
    return static_cast<type_real>(*this);
  }

  // Conversion to Omega (angular frequency)
  explicit inline operator Omega() const;

  // Conversion to Seconds
  explicit inline operator Seconds() const;
};

} // namespace specfem::datatype
