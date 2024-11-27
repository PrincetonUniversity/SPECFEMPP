#pragma once

namespace specfem {
namespace plotter {
/**
 * @brief Base writer class
 *
 */
class plotter {
public:
  /**
   * @brief Construct a new plotter object
   *
   * @param time_interval Time interval between subsequent plots
   */
  plotter(const int time_interval) : time_interval(time_interval){};

  /**
   * @brief Method to plot the data
   *
   */
  virtual void plot(){};

  /**
   * @brief Returns true if the data should be plotted at the current
   * timestep. Updates the internal timestep counter
   *
   * @param istep Current timestep
   * @return true if the data should be plotted at the current timestep
   */
  bool should_plot(const int istep) {
    if (istep % time_interval == 0) {
      this->m_istep = istep;
      return true;
    }
    return false;
  }

protected:
  int time_interval;
  int m_istep;
};

} // namespace plotter
} // namespace specfem
