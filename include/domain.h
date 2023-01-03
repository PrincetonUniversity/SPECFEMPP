#ifndef DOMAIN_H
#define DOMAIN_H

#include "../include/compute.h"
#include "../include/config.h"
#include "../include/quadrature.h"
#include <Kokkos_Core.hpp>

namespace specfem {
namespace Domain {
class Domain {

public:
  virtual specfem::HostView2d<type_real> get_field() const {
    return this->field;
  }
  virtual specfem::HostView2d<type_real> get_field_dot() const {
    return this->field_dot;
  }
  virtual specfem::HostView2d<type_real> get_field_dot_dot() const {
    return this->field_dot_dot;
  }
  virtual specfem::HostView2d<type_real> get_rmass_inverse() const {
    return this->rmass_inverse;
  }

  virtual void compute_forces(){};

private:
  specfem::HostView2d<type_real> field;
  specfem::HostView2d<type_real> field_dot;
  specfem::HostView2d<type_real> field_dot_dot;
  specfem::HostView2d<type_real> rmass_inverse;
};

class Elastic : public Domain {
public:
  /**
   * @brief Get a view of displacement
   *
   * @return specfem::HostView2d<type_real> View of the displacement
   */
  specfem::HostView2d<type_real> get_field() const override {
    return this->field;
  }
  /**
   * @brief Get a view of velocity
   *
   * @return specfem::HostView2d<type_real> view of the velocity
   */
  specfem::HostView2d<type_real> get_field_dot() const override {
    return this->field_dot;
  }
  /**
   * @brief Get a view of acceleration
   *
   * @return specfem::HostView2d<type_real> view of the acceleration
   */
  specfem::HostView2d<type_real> get_field_dot_dot() const override {
    return this->field_dot_dot;
  }

  specfem::HostView2d<type_real> get_rmass_inverse() const override {
    return this->rmass_inverse;
  }

  /**
   * @brief Construct a new Elastic domain object
   *
   * @param ndim Number of dimensions
   * @param nglob Total number of distinct quadrature points inside the domain
   * @param material_properties Pointer to specfem::compute::properties struct
   * used to store material properties
   * @param quadx Pointer to quadrature object in x-dimension
   * @param quadx Pointer to quadrature object in z-dimension
   */
  Elastic(const int ndim, const int nglob, specfem::compute::compute *compute,
          specfem::compute::properties *material_properties,
          specfem::compute::partial_derivatives *partial_derivatives,
          quadrature::quadrature *quadx, quadrature::quadrature *quadz);

  void compute_forces() override;

private:
  specfem::HostView2d<type_real> field; ///< Displacement inside elastic domain
  specfem::HostView2d<type_real> field_dot; ///< Velocity inside elastic domain
  specfem::HostView2d<type_real> field_dot_dot; ///< Acceleration inside elastic
                                                ///< domain
  specfem::HostView2d<type_real> rmass_inverse; ///< Inverse of mass matrix
                                                ///< inside elastic domain
  specfem::compute::compute *compute;
  specfem::compute::properties *material_properties; ///< Pointer to struct used
                                                     ///< to store material
                                                     ///< properties
  specfem::compute::partial_derivatives *partial_derivatives;
  quadrature::quadrature *quadx; ///< Pointer to quadrature object in
                                 ///< x-dimension
  quadrature::quadrature *quadz; ///< Pointer to quadrature object in
                                 ///< z-dimension
  int nelem_domain;              ///< Total number of elements in this domain
  specfem::HostView1d<int> ispec_domain; ///< Array containing global
                                         ///< indices(ispec) of all elements in
                                         ///< this domain
};
} // namespace Domain
} // namespace specfem

#endif
