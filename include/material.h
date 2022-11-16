#ifndef MATERIAL_H
#define MATERIAL_H

#include "../include/config.h"
#include "../include/utils.h"
#include <ostream>

namespace specfem {

/**
 * @brief Base material class
 *
 */
class material {
public:
  /**
   * @brief Construct a new material object
   *
   */
  material();
  /**
   * @brief Virtual function to assign values read from database file to
   * material class members
   *
   * @param holder holder used to hold read values
   */
  virtual void assign(utilities::value_holder &holder){};
};

class elastic_material : public material {
public:
  /**
   * @brief Construct a new elastic material object
   *
   */
  elastic_material();
  /**
   * @brief Assign elastic material values
   *
   * @param holder holder used to hold read values
   */
  void assign(utilities::value_holder &holder) override;
  friend std::ostream &operator<<(std::ostream &out, const elastic_material &h);

private:
  /**
   * @brief Elastic material properties
   *
   */
  type_real density, cs, cp, Qkappa, Qmu, compaction_grad, lambdaplus2mu, mu,
      lambda, kappa, young, poisson;
};

class acoustic_material : public material {
public:
  /**
   * @brief Construct a new acoustic material object
   *
   */
  acoustic_material();
  /**
   * @brief Assign acoustic material values
   *
   * @param holder holder used to hold read values
   */
  void assign(utilities::value_holder &holder) override;
  friend std::ostream &operator<<(std::ostream &out,
                                  const acoustic_material &h);

private:
  /**
   * @brief Acoustic material properties
   *
   */
  type_real density, cs, cp, Qkappa, Qmu, compaction_grad, lambdaplus2mu, mu,
      lambda, kappa, young, poisson;
};

std::ostream &operator<<(std::ostream &out, const specfem::elastic_material &h);

std::ostream &operator<<(std::ostream &out,
                         const specfem::acoustic_material &h);
} // namespace specfem

#endif
