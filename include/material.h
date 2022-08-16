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
  void assign(utilities::value_holder &holder);
  friend std::ostream &operator<<(std::ostream &out, const elastic_material &h);

private:
  /*! \var type_real density
      \brief Density of the elastic material
  */

  /*! \var type_real cs
      \brief of the elastic material
  */
  type_real density, cs, cp, Qkappa, Qmu, compaction_grad, lambdaplus2mu, mu,
      lambda, kappa, young, poisson;
};

class acoustic_material : public material {
public:
  acoustic_material();
  void assign(utilities::value_holder &holder);
  friend std::ostream &operator<<(std::ostream &out,
                                  const acoustic_material &h);

private:
  type_real density, cs, cp, Qkappa, Qmu, compaction_grad, lambdaplus2mu, mu,
      lambda, kappa, young, poisson;
};

std::ostream &operator<<(std::ostream &out, const specfem::elastic_material &h);

std::ostream &operator<<(std::ostream &out,
                         const specfem::acoustic_material &h);

} // namespace specfem

#endif
