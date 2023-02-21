#ifndef MATERIAL_H
#define MATERIAL_H

#include "../include/config.h"
#include "../include/specfem_mpi.h"
#include "../include/utils.h"
#include <ostream>
#include <tuple>

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
  virtual void assign(utilities::input_holder &holder){};
  virtual utilities::return_holder get_properties() {
    utilities::return_holder holder{};
    return holder;
  };
  virtual element_type get_ispec_type() {
    element_type dummy{ elastic };
    return dummy;
  };

  virtual std::string print() const { return ""; }
};

/**
 * @brief Elastic material class
 *
 * Defines the routines required to read and assign elastic material properties
 * to specfem mesh.
 */
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
  void assign(utilities::input_holder &holder) override;
  /**
   * @brief User output
   * Prints the read material values and additional information on
   * console/output file
   *
   * @param out Empty output stream
   * @param h elastic material holder to pass read values
   * @return std::ostream& Output stream to be displayed
   */
  friend std::ostream &operator<<(std::ostream &out, const elastic_material &h);
  /**
   * @brief Get private elastic material properties
   *
   * @return utilities::return_holder holder used to return elastic material
   * properties
   */
  utilities::return_holder get_properties() override;
  element_type get_ispec_type() { return ispec_type; };

  std::string print() const override;

private:
  /**
   * @name Elastic material properties
   *
   */
  ///@{
  type_real density;
  type_real cs;
  type_real cp;
  type_real Qkappa;
  type_real Qmu;
  type_real compaction_grad;
  type_real lambdaplus2mu;
  type_real mu;
  type_real lambda;
  type_real kappa;
  type_real young;
  type_real poisson;
  ///@}
  element_type ispec_type = elastic; ///< Type or element == specfem::elastic
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
  void assign(utilities::input_holder &holder) override;
  friend std::ostream &operator<<(std::ostream &out,
                                  const acoustic_material &h);
  element_type get_ispec_type() { return ispec_type; };
  std::string print() const override;

private:
  /**
   * @brief Acoustic material properties
   *
   */
  type_real density, cs, cp, Qkappa, Qmu, compaction_grad, lambdaplus2mu, mu,
      lambda, kappa, young, poisson;
  element_type ispec_type = acoustic;
};

std::ostream &operator<<(std::ostream &out, const specfem::elastic_material &h);

std::ostream &operator<<(std::ostream &out,
                         const specfem::acoustic_material &h);

} // namespace specfem

#endif
