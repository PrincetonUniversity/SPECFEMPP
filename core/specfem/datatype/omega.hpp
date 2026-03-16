#pragma once

namespace specfem::datatype {

// Forward declarations

class Hertz;
class Seconds;

class Omega : public type_real {
public:
  using type_real::type_real; // Inherit constructors

  // Allow implicit conversion to type_real
  explicit inline operator type_real() const {
    return static_cast<type_real>(*this);
  }

  explicit inline operator Hertz() const;

  explicit inline operator Seconds() const;
};

} // namespace specfem::datatype
