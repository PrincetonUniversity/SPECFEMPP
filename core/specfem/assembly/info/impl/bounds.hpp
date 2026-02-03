#pragma once

#include "specfem_setup.hpp"

namespace specfem::assembly::info::impl {

struct Bounds {
public:
  type_real min;
  type_real max;

  Bounds() : min(0), max(0) {}

  Bounds(type_real min_in, type_real max_in)
      : min(min_in),
        max(max_in) {}

  type_real length() const { return this->max - this->min; }
  type_real ratio() const { 
    if (this->min == 0) {
      throw std::runtime_error("Bounds::ratio(): min is zero, cannot compute ratio.");
    }
    return this->max / this->min; }
  type_real center() const { return 0.5 * (this->max + this->min); }

  Bounds&operator=(const type_real value) {
    this->min = value;
    this->max = value;
    return *this;
  }
};

} // namespace specfem::assembly::info::impl  