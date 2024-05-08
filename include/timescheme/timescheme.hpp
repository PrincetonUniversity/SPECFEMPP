#ifndef _TIMESCHEME_HPP
#define _TIMESCHEME_HPP

#include "domain/interface.hpp"
#include "specfem_setup.hpp"
#include <ostream>

namespace specfem {
namespace time_scheme {

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

/**
 * @brief Base time scheme class.
 *
 */
class time_scheme {
public:
  time_scheme(const int nstep, const int nstep_between_samples,
              const type_real dt)
      : nstep(nstep), nstep_between_samples(nstep_between_samples),
        seismogram_timestep(0), dt(dt) {}

  ForwardRange iterate_forward() { return ForwardRange(nstep, dt); }
  BackwardRange iterate_backward() { return BackwardRange(nstep, dt); }
  int get_max_timestep() { return nstep; }
  void increment_seismogram_step() { seismogram_timestep++; }
  bool compute_seismogram(const int istep) const {
    return (istep % nstep_between_samples == 0);
  }
  int get_seismogram_step() const { return seismogram_timestep; }

  virtual void
  apply_predictor_phase_forward(const specfem::element::medium_tag tag) = 0;

  virtual void
  apply_corrector_phase_forward(const specfem::element::medium_tag tag) = 0;

  virtual void
  apply_predictor_phase_backward(const specfem::element::medium_tag tag) = 0;

  virtual void
  apply_corrector_phase_backward(const specfem::element::medium_tag tag) = 0;

  virtual void link_assembly(const specfem::compute::assembly &assembly) = 0;

  virtual specfem::enums::time_scheme::type timescheme() const = 0;

  ~time_scheme() = default;

  virtual void print(std::ostream &out) const = 0;

  int get_max_seismogram_step() const { return nstep / nstep_between_samples; }

  virtual type_real get_timestep() const = 0;

private:
  int nstep;
  int seismogram_timestep;
  int nstep_between_samples;
  type_real dt;
};

// /**
//  * @brief Base time scheme class.
//  *
//  */
// class TimeScheme {

// public:
//   /**
//    * @brief Get the timescheme type
//    *
//    */
//   virtual specfem::enums::time_scheme::type timescheme() const = 0;
//   /**
//    * @brief Return the status of simulation
//    *
//    * @return false if current step >= number of steps
//    * @return true if current step < number of steps
//    */
//   virtual bool status() const { return false; };
//   /**
//    * @brief increment by one timestep, also updates the simulation time by dt
//    *
//    */
//   virtual void increment_time(){};
//   /**
//    * @brief Get the current simulation time
//    *
//    * @return type_real current time
//    */
//   virtual type_real get_time() const { return 0.0; }
//   /**
//    * @brief Get the current timestep
//    *
//    * @return int current timestep
//    */
//   virtual int get_timestep() const { return 0; }
//   /**
//    * @brief reset current time to t0 and timestep to 0
//    *
//    */
//   virtual void reset_time(){};
//   /**
//    * @brief Get the max timestep (nstep) of the simuation
//    *
//    * @return int max timestep
//    */
//   virtual int get_max_timestep() { return 0; }
//   /**
//    * @brief Apply predictor phase of the timescheme
//    *
//    * @param domain_class Pointer to domain class to apply predictor phase
//    */
//   virtual void apply_predictor_phase(
//       specfem::kokkos::DeviceView2d<type_real, Kokkos::LayoutLeft> field,
//       specfem::kokkos::DeviceView2d<type_real, Kokkos::LayoutLeft> field_dot,
//       specfem::kokkos::DeviceView2d<type_real, Kokkos::LayoutLeft>
//           field_dot_dot){};
//   /**
//    * @brief Apply corrector phase of the timescheme
//    *
//    * @param domain_class Pointer to domain class to apply corrector phase
//    */
//   virtual void apply_corrector_phase(
//       specfem::kokkos::DeviceView2d<type_real, Kokkos::LayoutLeft> field,
//       specfem::kokkos::DeviceView2d<type_real, Kokkos::LayoutLeft> field_dot,
//       specfem::kokkos::DeviceView2d<type_real, Kokkos::LayoutLeft>
//           field_dot_dot){};

//   friend std::ostream &operator<<(std::ostream &out, TimeScheme &ts);
//   /**
//    * @brief Log timescheme information to console
//    */
//   virtual void print(std::ostream &out) const;
//   /**
//    * @brief Compute if seismogram needs to be calculated at this timestep
//    *
//    */
//   virtual bool compute_seismogram() const { return false; }
//   /**
//    * @brief Get the current seismogram step
//    *
//    * @return int value of the current seismogram step
//    */
//   virtual int get_seismogram_step() const { return 0; }
//   /**
//    * @brief Get the max seismogram step
//    *
//    * @return int maximum value of seismogram step
//    */
//   virtual int get_max_seismogram_step() const { return 0; }
//   /**
//    * @brief increment seismogram step
//    *
//    */
//   virtual void increment_seismogram_step(){};
//   /**
//    * @brief Get time increment
//    *
//    */
//   virtual type_real get_time_increment() const { return 0.0; }

//   /**
//    * @brief Default destructor
//    *
//    */
//   virtual ~TimeScheme() = default;
// };

std::ostream &operator<<(std::ostream &out,
                         specfem::time_scheme::time_scheme &ts);
} // namespace time_scheme
} // namespace specfem
#endif
