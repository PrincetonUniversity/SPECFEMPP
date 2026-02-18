#pragma once
#include "specfem/setup.hpp"

namespace specfem::time_scheme::impl {

/**
 * @brief Iterator for backward time integration
 *
 * Iterates from the last timestep down to zero, yielding (timestep, dt) pairs.
 */
class BackwardIterator {
public:
  BackwardIterator(int value, type_real dt) : value(value), dt(dt) {}

  std::tuple<int, type_real> operator*() const { return { value, dt }; }

  BackwardIterator &operator++() {
    value--;
    return *this;
  }

  bool operator!=(const BackwardIterator &other) const {
    const auto [other_value, other_dt] = *other;
    return value != other_value;
  }

private:
  int value;
  type_real dt;
};

/**
 * @brief Iterator for forward time integration
 *
 * Iterates from timestep zero to the maximum, yielding (timestep, dt) pairs.
 */
class ForwardIterator {
public:
  ForwardIterator(int value, type_real dt) : value(value), dt(dt) {}

  std::tuple<int, type_real> operator*() const { return { value, dt }; }

  ForwardIterator &operator++() {
    value++;
    return *this;
  }

  bool operator!=(const ForwardIterator &other) const {
    const auto [other_value, other_dt] = *other;
    return value != other_value;
  }

private:
  int value;
  type_real dt;
};

/**
 * @brief Range for forward time iteration
 */
class ForwardRange {
public:
  ForwardRange(int nsteps, const type_real dt)
      : start_(0), end_(nsteps), dt(dt) {}
  ForwardIterator begin() const { return ForwardIterator(start_, dt); }
  ForwardIterator end() const { return ForwardIterator(end_, dt); }

private:
  int start_;
  int end_;
  type_real dt;
};

/**
 * @brief Range for backward time iteration
 */
class BackwardRange {
public:
  BackwardRange(int nsteps, const type_real dt)
      : start_(nsteps - 1), end_(-1), dt(dt) {}
  BackwardIterator begin() const { return BackwardIterator(start_, dt); }
  BackwardIterator end() const { return BackwardIterator(end_, dt); }

private:
  int start_;
  int end_;
  type_real dt;
};

} // namespace specfem::time_scheme::impl
