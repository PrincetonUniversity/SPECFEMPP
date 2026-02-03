#pragma once

#include "specfem_setup.hpp"
#include "yaml-cpp/yaml.h"
#include <ctime>
#include <tuple>

namespace specfem {
namespace runtime_configuration {
/**
 * @brief Header class to store and print the title and description of the
 * simulation
 *
 */
class header {

public:
  /**
   * @brief Construct a new header object
   *
   * @param title Title of simulation
   * @param description Description of the simulation
   */
  header(std::string title, std::string description)
      : title(title), description(description) {};
  /**
   * @brief Construct a new header object using YAML node
   *
   * @param Node YAML node as read from a YAML file
   */
  header(const YAML::Node &Node);

  /**
   * @brief Get the title
   *
   * @return std::string title of the simulation
   */
  std::string get_title() const { return this->title; }
  /**
   * @brief Get the description
   *
   * @return std::string description of the simulation
   */
  std::string get_description() const { return this->description; }

  friend std::ostream &operator<<(std::ostream &out, header &header);

private:
  std::string title;       ///< Title of the simulation
  std::string description; ///< Description of the simulation
};
} // namespace runtime_configuration
} // namespace specfem
